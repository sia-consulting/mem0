import json
import logging

from mem0.memory.utils import format_entities, remove_spaces_from_entities

try:
    from langchain_neo4j import Neo4jGraph
except ImportError:
    raise ImportError("langchain_neo4j is not installed. Please install it using pip install langchain-neo4j")

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    raise ImportError("rank_bm25 is not installed. Please install it using pip install rank-bm25")

from mem0.graphs.tools import (
    DELETE_MEMORY_STRUCT_TOOL_GRAPH,
    DELETE_MEMORY_TOOL_GRAPH,
    EXTRACT_ENTITIES_STRUCT_TOOL,
    EXTRACT_ENTITIES_TOOL,
    RELATIONS_STRUCT_TOOL,
    RELATIONS_TOOL,
)
from mem0.graphs.utils import EXTRACT_RELATIONS_PROMPT, get_delete_messages
from mem0.utils.factory import EmbedderFactory, LlmFactory

logger = logging.getLogger(__name__)

# Keys reserved for internal graph management — filtered from user-facing property results
# and protected from overwrites in _sanitize_properties().
_SYSTEM_RESERVED_KEYS = frozenset({
    "name", "user_id", "agent_id", "run_id", "embedding", "mentions",
    "created", "source", "valid", "created_at", "updated_at",
    "invalidated_at", "relationship", "is_anchor", "entity_type",
})


def _parse_entity_properties(item):
    """Parse properties from an entity extraction result.

    Handles both non-struct tools (``properties`` is a dict) and struct tools
    (``properties_json`` is a JSON string).
    """
    props = item.get("properties")
    if isinstance(props, dict):
        return {k: str(v) for k, v in props.items()}
    props_json = item.get("properties_json")
    if props_json and isinstance(props_json, str):
        try:
            parsed = json.loads(props_json)
            if isinstance(parsed, dict):
                return {k: str(v) for k, v in parsed.items()}
        except (json.JSONDecodeError, ValueError):
            pass
    return {}


def _parse_relation_properties(item):
    """Parse properties from a relation extraction result.

    Handles both non-struct tools (``properties`` is a dict) and struct tools
    (``properties_json`` is a JSON string).
    """
    props = item.get("properties")
    if isinstance(props, dict):
        return {k: str(v) for k, v in props.items()}
    props_json = item.get("properties_json")
    if props_json and isinstance(props_json, str):
        try:
            parsed = json.loads(props_json)
            if isinstance(parsed, dict):
                return {k: str(v) for k, v in parsed.items()}
        except (json.JSONDecodeError, ValueError):
            pass
    return {}


def _sanitize_properties(props):
    """Sanitize a properties dict for safe storage in Neo4j.

    Ensures all keys are valid Python identifiers and values are flat strings.
    Filters out internal/reserved keys (see ``_SYSTEM_RESERVED_KEYS``).

    All values are coerced to strings because Neo4j property maps passed
    via ``SET n += $props`` must have homogeneous types within the map,
    and string is the safest common denominator for arbitrary LLM-extracted
    data.  Callers should be aware that numeric or boolean values will be
    stored as their string representations.
    """
    if not props:
        return {}
    sanitized = {}
    for k, v in props.items():
        key = k.lower().replace(" ", "_").replace("-", "_")
        if key in _SYSTEM_RESERVED_KEYS:
            continue
        if not key.isidentifier():
            continue
        sanitized[key] = str(v) if v is not None else ""
    return sanitized


class MemoryGraph:
    def __init__(self, config):
        self.config = config
        self.graph = Neo4jGraph(
            url=self.config.graph_store.config.url,
            username=self.config.graph_store.config.username,
            password=self.config.graph_store.config.password,
            database=self.config.graph_store.config.database,
            refresh_schema=False,
            driver_config={"notifications_min_severity": "OFF"},
        )
        self.embedding_model = EmbedderFactory.create(
            self.config.embedder.provider, self.config.embedder.config, self.config.vector_store.config
        )
        self.node_label = ":`__Entity__`" if self.config.graph_store.config.base_label else ""

        if self.config.graph_store.config.base_label:
            # Safely add user_id index
            try:
                self.graph.query(f"CREATE INDEX entity_single IF NOT EXISTS FOR (n {self.node_label}) ON (n.user_id)")
            except Exception:
                pass
            try:  # Safely try to add composite index (Enterprise only)
                self.graph.query(
                    f"CREATE INDEX entity_composite IF NOT EXISTS FOR (n {self.node_label}) ON (n.name, n.user_id)"
                )
            except Exception:
                pass

        # Default to openai if no specific provider is configured
        self.llm_provider = "openai"
        if self.config.llm and self.config.llm.provider:
            self.llm_provider = self.config.llm.provider
        if self.config.graph_store and self.config.graph_store.llm and self.config.graph_store.llm.provider:
            self.llm_provider = self.config.graph_store.llm.provider

        # Get LLM config with proper null checks
        llm_config = None
        if self.config.graph_store and self.config.graph_store.llm and hasattr(self.config.graph_store.llm, "config"):
            llm_config = self.config.graph_store.llm.config
        elif hasattr(self.config.llm, "config"):
            llm_config = self.config.llm.config
        self.llm = LlmFactory.create(self.llm_provider, llm_config)
        self.user_id = None
        # Use threshold from graph_store config, default to 0.7 for backward compatibility
        self.threshold = self.config.graph_store.threshold if hasattr(self.config.graph_store, 'threshold') else 0.7
        # Anchor node name per agent/user scope (Phase 2)
        self.anchor_node_name = getattr(self.config.graph_store, 'anchor_node_name', 'me')

    def add(self, data, filters):
        """
        Adds data to the graph.

        Args:
            data (str): The data to add to the graph.
            filters (dict): A dictionary containing filters to be applied during the addition.
        """
        # Ensure the anchor "Me" node exists before processing entities (Phase 2)
        self._ensure_me_node(filters)

        entity_type_map = self._retrieve_nodes_from_data(data, filters)
        to_be_added = self._establish_nodes_relations_from_data(data, filters, entity_type_map)
        search_output = self._search_graph_db(node_list=list(entity_type_map.keys()), filters=filters)
        to_be_deleted = self._get_delete_entities_from_search_output(search_output, data, filters)

        # TODO: Batch queries with APOC plugin
        # TODO: Add more filter support
        deleted_entities = self._delete_entities(to_be_deleted, filters)
        added_entities = self._add_entities(to_be_added, filters, entity_type_map)

        # Connect any orphan nodes to the anchor "Me" node (Phase 2)
        self._connect_orphans_to_me(to_be_added, filters)

        return {"deleted_entities": deleted_entities, "added_entities": added_entities}

    def search(self, query, filters, limit=100):
        """
        Search for memories and related graph data.

        Args:
            query (str): Query to search for.
            filters (dict): A dictionary containing filters to be applied during the search.
            limit (int): The maximum number of nodes and relationships to retrieve. Defaults to 100.

        Returns:
            list: A list of dicts, each containing source, relationship, destination
                  and optionally source_properties, edge_properties, destination_properties.
        """
        entity_type_map = self._retrieve_nodes_from_data(query, filters)
        search_output = self._search_graph_db(node_list=list(entity_type_map.keys()), filters=filters)

        if not search_output:
            return []

        search_outputs_sequence = [
            [item["source"], item["relationship"], item["destination"]] for item in search_output
        ]
        bm25 = BM25Okapi(search_outputs_sequence)

        tokenized_query = query.split(" ")
        reranked_results = bm25.get_top_n(tokenized_query, search_outputs_sequence, n=5)

        # Build a lookup from (source, rel, dest) → original search item for properties
        props_lookup = {}
        for item in search_output:
            key = (item["source"], item["relationship"], item["destination"])
            if key not in props_lookup:
                props_lookup[key] = item

        search_results = []
        for item in reranked_results:
            entry = {"source": item[0], "relationship": item[1], "destination": item[2]}
            # Attach filtered properties if available
            original = props_lookup.get((item[0], item[1], item[2]))
            if original:
                src_props = {k: v for k, v in (original.get("source_properties") or {}).items() if k not in _SYSTEM_RESERVED_KEYS}
                edge_props = {k: v for k, v in (original.get("edge_properties") or {}).items() if k not in _SYSTEM_RESERVED_KEYS}
                dest_props = {k: v for k, v in (original.get("destination_properties") or {}).items() if k not in _SYSTEM_RESERVED_KEYS}
                if src_props:
                    entry["source_properties"] = src_props
                if edge_props:
                    entry["edge_properties"] = edge_props
                if dest_props:
                    entry["destination_properties"] = dest_props
            search_results.append(entry)

        logger.info(f"Returned {len(search_results)} search results")

        return search_results

    def delete(self, data, filters):
        """
        Delete graph entities associated with the given memory text.

        Extracts entities and relationships from the memory text using the same
        pipeline as add(), then soft-deletes the matching relationships in the graph.

        Args:
            data (str): The memory text whose graph entities should be removed.
            filters (dict): Scope filters (user_id, agent_id, run_id).
        """
        try:
            entity_type_map = self._retrieve_nodes_from_data(data, filters)
            if not entity_type_map:
                logger.debug("No entities found in memory text, skipping graph cleanup")
                return
            to_be_deleted = self._establish_nodes_relations_from_data(data, filters, entity_type_map)
            if to_be_deleted:
                self._delete_entities(to_be_deleted, filters)
        except Exception as e:
            logger.error(f"Error during graph cleanup for memory delete: {e}")

    def delete_all(self, filters):
        # Build node properties for filtering
        node_props = ["user_id: $user_id"]
        if filters.get("agent_id"):
            node_props.append("agent_id: $agent_id")
        if filters.get("run_id"):
            node_props.append("run_id: $run_id")
        node_props_str = ", ".join(node_props)

        cypher = f"""
        MATCH (n {self.node_label} {{{node_props_str}}})
        DETACH DELETE n
        """
        params = {"user_id": filters["user_id"]}
        if filters.get("agent_id"):
            params["agent_id"] = filters["agent_id"]
        if filters.get("run_id"):
            params["run_id"] = filters["run_id"]
        self.graph.query(cypher, params=params)

    def get_all(self, filters, limit=100):
        """
        Retrieves all nodes and relationships from the graph database based on optional filtering criteria.
         Args:
            filters (dict): A dictionary containing filters to be applied during the retrieval.
            limit (int): The maximum number of nodes and relationships to retrieve. Defaults to 100.
        Returns:
            list: A list of dictionaries, each containing:
                - 'contexts': The base data store response for each memory.
                - 'entities': A list of strings representing the nodes and relationships
        """
        params = {"user_id": filters["user_id"], "limit": limit}

        # Build node properties based on filters
        node_props = ["user_id: $user_id"]
        if filters.get("agent_id"):
            node_props.append("agent_id: $agent_id")
            params["agent_id"] = filters["agent_id"]
        if filters.get("run_id"):
            node_props.append("run_id: $run_id")
            params["run_id"] = filters["run_id"]
        node_props_str = ", ".join(node_props)

        query = f"""
        MATCH (n {self.node_label} {{{node_props_str}}})-[r]->(m {self.node_label} {{{node_props_str}}})
        WHERE r.valid IS NULL OR r.valid = true
        RETURN n.name AS source, type(r) AS relationship, m.name AS target,
               properties(n) AS source_properties, properties(r) AS edge_properties, properties(m) AS target_properties
        LIMIT $limit
        """
        results = self.graph.query(query, params=params)

        final_results = []
        for result in results:
            entry = {
                "source": result["source"],
                "relationship": result["relationship"],
                "target": result["target"],
            }
            # Include custom properties, filtering out system keys
            src_props = {k: v for k, v in (result.get("source_properties") or {}).items() if k not in _SYSTEM_RESERVED_KEYS}
            edge_props = {k: v for k, v in (result.get("edge_properties") or {}).items() if k not in _SYSTEM_RESERVED_KEYS}
            tgt_props = {k: v for k, v in (result.get("target_properties") or {}).items() if k not in _SYSTEM_RESERVED_KEYS}
            if src_props:
                entry["source_properties"] = src_props
            if edge_props:
                entry["edge_properties"] = edge_props
            if tgt_props:
                entry["target_properties"] = tgt_props
            final_results.append(entry)

        logger.info(f"Retrieved {len(final_results)} relationships")

        return final_results

    def _retrieve_nodes_from_data(self, data, filters):
        """Extracts all the entities mentioned in the query, including their properties."""
        _tools = [EXTRACT_ENTITIES_TOOL]
        if self.llm_provider in ["azure_openai_structured", "openai_structured"]:
            _tools = [EXTRACT_ENTITIES_STRUCT_TOOL]
        search_results = self.llm.generate_response(
            messages=[
                {
                    "role": "system",
                    "content": f"You are a smart assistant who understands entities and their types in a given text. If user message contains self reference such as 'I', 'me', 'my' etc. then use {filters['user_id']} as the source entity. Extract all the entities from the text along with any notable properties (key-value attributes) for each entity. ***DO NOT*** answer the question itself if the given text is a question.",
                },
                {"role": "user", "content": data},
            ],
            tools=_tools,
        )

        entity_type_map = {}

        try:
            for tool_call in search_results["tool_calls"]:
                if tool_call["name"] != "extract_entities":
                    continue
                for item in tool_call.get("arguments", {}).get("entities", []):
                    entity_name = item["entity"]
                    entity_type = item["entity_type"]
                    # Parse properties: from dict (non-struct) or JSON string (struct)
                    properties = _parse_entity_properties(item)
                    entity_type_map[entity_name] = {
                        "type": entity_type,
                        "properties": properties,
                    }
        except Exception as e:
            logger.exception(
                f"Error in search tool: {e}, llm_provider={self.llm_provider}, search_results={search_results}"
            )

        entity_type_map = {
            k.lower().replace(" ", "_"): {
                "type": v["type"].lower().replace(" ", "_"),
                "properties": v.get("properties", {}),
            }
            for k, v in entity_type_map.items()
        }
        logger.debug(f"Entity type map: {entity_type_map}\n search_results={search_results}")
        return entity_type_map

    def _establish_nodes_relations_from_data(self, data, filters, entity_type_map):
        """Establish relations among the extracted nodes."""

        # Compose user identification string for prompt
        user_identity = f"user_id: {filters['user_id']}"
        if filters.get("agent_id"):
            user_identity += f", agent_id: {filters['agent_id']}"
        if filters.get("run_id"):
            user_identity += f", run_id: {filters['run_id']}"

        if self.config.graph_store.custom_prompt:
            system_content = EXTRACT_RELATIONS_PROMPT.replace("USER_ID", user_identity)
            # Add the custom prompt line if configured
            system_content = system_content.replace("CUSTOM_PROMPT", f"5. {self.config.graph_store.custom_prompt}")
            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": data},
            ]
        else:
            system_content = EXTRACT_RELATIONS_PROMPT.replace("USER_ID", user_identity)
            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": f"List of entities: {list(entity_type_map.keys())}. \n\nText: {data}"},
            ]

        _tools = [RELATIONS_TOOL]
        if self.llm_provider in ["azure_openai_structured", "openai_structured"]:
            _tools = [RELATIONS_STRUCT_TOOL]

        extracted_entities = self.llm.generate_response(
            messages=messages,
            tools=_tools,
        )

        entities = []
        if extracted_entities.get("tool_calls"):
            raw_entities = extracted_entities["tool_calls"][0].get("arguments", {}).get("entities", [])
            # Parse relationship properties from each entity
            for item in raw_entities:
                if isinstance(item, dict):
                    item["edge_properties"] = _parse_relation_properties(item)
                    # Remove the raw properties fields to avoid confusion downstream
                    item.pop("properties", None)
                    item.pop("properties_json", None)
            entities = raw_entities

        entities = self._remove_spaces_from_entities(entities)
        logger.debug(f"Extracted entities: {entities}")
        return entities

    def _search_graph_db(self, node_list, filters, limit=100):
        """Search similar nodes among and their respective incoming and outgoing relations.

        Returns properties on source nodes, edges, and destination nodes when available.
        """
        result_relations = []

        # Build node properties for filtering
        node_props = ["user_id: $user_id"]
        if filters.get("agent_id"):
            node_props.append("agent_id: $agent_id")
        if filters.get("run_id"):
            node_props.append("run_id: $run_id")
        node_props_str = ", ".join(node_props)

        for node in node_list:
            n_embedding = self.embedding_model.embed(node)

            cypher_query = f"""
            MATCH (n {self.node_label} {{{node_props_str}}})
            WHERE n.embedding IS NOT NULL
            WITH n, round(2 * vector.similarity.cosine(n.embedding, $n_embedding) - 1, 4) AS similarity // denormalize for backward compatibility
            WHERE similarity >= $threshold
            CALL {{
                WITH n
                MATCH (n)-[r]->(m {self.node_label} {{{node_props_str}}})
                WHERE r.valid IS NULL OR r.valid = true
                RETURN n.name AS source, elementId(n) AS source_id, type(r) AS relationship, elementId(r) AS relation_id, m.name AS destination, elementId(m) AS destination_id,
                       properties(n) AS source_properties, properties(r) AS edge_properties, properties(m) AS destination_properties
                UNION
                WITH n
                MATCH (n)<-[r]-(m {self.node_label} {{{node_props_str}}})
                WHERE r.valid IS NULL OR r.valid = true
                RETURN m.name AS source, elementId(m) AS source_id, type(r) AS relationship, elementId(r) AS relation_id, n.name AS destination, elementId(n) AS destination_id,
                       properties(m) AS source_properties, properties(r) AS edge_properties, properties(n) AS destination_properties
            }}
            WITH distinct source, source_id, relationship, relation_id, destination, destination_id, similarity,
                 source_properties, edge_properties, destination_properties
            RETURN source, source_id, relationship, relation_id, destination, destination_id, similarity,
                   source_properties, edge_properties, destination_properties
            ORDER BY similarity DESC
            LIMIT $limit
            """

            params = {
                "n_embedding": n_embedding,
                "threshold": self.threshold,
                "user_id": filters["user_id"],
                "limit": limit,
            }
            if filters.get("agent_id"):
                params["agent_id"] = filters["agent_id"]
            if filters.get("run_id"):
                params["run_id"] = filters["run_id"]

            ans = self.graph.query(cypher_query, params=params)
            result_relations.extend(ans)

        return result_relations

    def _get_delete_entities_from_search_output(self, search_output, data, filters):
        """Get the entities to be deleted from the search output."""
        search_output_string = format_entities(search_output)

        # Compose user identification string for prompt
        user_identity = f"user_id: {filters['user_id']}"
        if filters.get("agent_id"):
            user_identity += f", agent_id: {filters['agent_id']}"
        if filters.get("run_id"):
            user_identity += f", run_id: {filters['run_id']}"

        system_prompt, user_prompt = get_delete_messages(search_output_string, data, user_identity)

        _tools = [DELETE_MEMORY_TOOL_GRAPH]
        if self.llm_provider in ["azure_openai_structured", "openai_structured"]:
            _tools = [
                DELETE_MEMORY_STRUCT_TOOL_GRAPH,
            ]

        memory_updates = self.llm.generate_response(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            tools=_tools,
        )

        to_be_deleted = []
        for item in memory_updates.get("tool_calls", []):
            if item.get("name") == "delete_graph_memory":
                to_be_deleted.append(item.get("arguments"))
        # Clean entities formatting
        to_be_deleted = self._remove_spaces_from_entities(to_be_deleted)
        logger.debug(f"Deleted relationships: {to_be_deleted}")
        return to_be_deleted

    def _delete_entities(self, to_be_deleted, filters):
        """Delete the entities from the graph."""
        user_id = filters["user_id"]
        agent_id = filters.get("agent_id", None)
        run_id = filters.get("run_id", None)
        results = []

        for item in to_be_deleted:
            source = item["source"]
            destination = item["destination"]
            relationship = item["relationship"]

            # Build the agent filter for the query

            params = {
                "source_name": source,
                "dest_name": destination,
                "user_id": user_id,
            }

            if agent_id:
                params["agent_id"] = agent_id
            if run_id:
                params["run_id"] = run_id

            # Build node properties for filtering
            source_props = ["name: $source_name", "user_id: $user_id"]
            dest_props = ["name: $dest_name", "user_id: $user_id"]
            if agent_id:
                source_props.append("agent_id: $agent_id")
                dest_props.append("agent_id: $agent_id")
            if run_id:
                source_props.append("run_id: $run_id")
                dest_props.append("run_id: $run_id")
            source_props_str = ", ".join(source_props)
            dest_props_str = ", ".join(dest_props)

            # Soft-delete: mark relationship as invalid instead of removing it,
            # enabling temporal reasoning over historical graph state.
            # See: https://github.com/mem0ai/mem0/issues/4187
            cypher = f"""
            MATCH (n {self.node_label} {{{source_props_str}}})
            -[r:{relationship}]->
            (m {self.node_label} {{{dest_props_str}}})
            WHERE r.valid IS NULL OR r.valid = true
            SET r.valid = false, r.invalidated_at = datetime()
            RETURN 
                n.name AS source,
                m.name AS target,
                type(r) AS relationship
            """

            result = self.graph.query(cypher, params=params)
            results.append(result)

        return results

    def _add_entities(self, to_be_added, filters, entity_type_map):
        """Add the new entities to the graph. Merge the nodes if they already exist.

        Supports rich properties on both nodes and edges.  ``entity_type_map``
        values may be plain strings (legacy) or dicts with ``type`` and
        ``properties`` keys (new rich format from Phase 1).
        """
        user_id = filters["user_id"]
        agent_id = filters.get("agent_id", None)
        run_id = filters.get("run_id", None)
        results = []
        for item in to_be_added:
            # entities
            source = item["source"]
            destination = item["destination"]
            relationship = item["relationship"]

            # Edge properties extracted alongside the relationship
            edge_props = _sanitize_properties(item.get("edge_properties", {}))

            # types — handle both legacy (str) and new (dict) entity_type_map values
            source_info = entity_type_map.get(source, "__User__")
            if isinstance(source_info, dict):
                source_type = source_info.get("type", "__User__")
                source_node_props = _sanitize_properties(source_info.get("properties", {}))
            else:
                source_type = source_info
                source_node_props = {}

            destination_info = entity_type_map.get(destination, "__User__")
            if isinstance(destination_info, dict):
                destination_type = destination_info.get("type", "__User__")
                dest_node_props = _sanitize_properties(destination_info.get("properties", {}))
            else:
                destination_type = destination_info
                dest_node_props = {}

            source_label = self.node_label if self.node_label else f":`{source_type}`"
            source_extra_set = f", source:`{source_type}`" if self.node_label else ""
            destination_label = self.node_label if self.node_label else f":`{destination_type}`"
            destination_extra_set = f", destination:`{destination_type}`" if self.node_label else ""

            # Cypher fragments for setting custom properties on edges
            edge_props_set = ", r += $edge_props" if edge_props else ""

            # embeddings
            source_embedding = self.embedding_model.embed(source)
            dest_embedding = self.embedding_model.embed(destination)

            # search for the nodes with the closest embeddings
            source_node_search_result = self._search_source_node(source_embedding, filters, threshold=self.threshold)
            destination_node_search_result = self._search_destination_node(dest_embedding, filters, threshold=self.threshold)

            # TODO: Create a cypher query and common params for all the cases
            if not destination_node_search_result and source_node_search_result:
                # Build destination MERGE properties
                merge_props = ["name: $destination_name", "user_id: $user_id"]
                if agent_id:
                    merge_props.append("agent_id: $agent_id")
                if run_id:
                    merge_props.append("run_id: $run_id")
                merge_props_str = ", ".join(merge_props)

                cypher = f"""
                MATCH (source)
                WHERE elementId(source) = $source_id
                SET source.mentions = coalesce(source.mentions, 0) + 1
                SET source += $source_node_props
                WITH source
                MERGE (destination {destination_label} {{{merge_props_str}}})
                ON CREATE SET
                    destination.created = timestamp(),
                    destination.mentions = 1
                    {destination_extra_set}
                ON MATCH SET
                    destination.mentions = coalesce(destination.mentions, 0) + 1
                WITH source, destination
                SET destination += $dest_node_props
                WITH source, destination
                CALL db.create.setNodeVectorProperty(destination, 'embedding', $destination_embedding)
                WITH source, destination
                MERGE (source)-[r:{relationship}]->(destination)
                ON CREATE SET
                    r.created_at = timestamp(),
                    r.updated_at = timestamp(),
                    r.mentions = 1,
                    r.valid = true
                    {edge_props_set}
                ON MATCH SET
                    r.mentions = coalesce(r.mentions, 0) + 1,
                    r.valid = true,
                    r.updated_at = timestamp(),
                    r.invalidated_at = null
                    {edge_props_set}
                RETURN source.name AS source, type(r) AS relationship, destination.name AS target
                """

                params = {
                    "source_id": source_node_search_result[0]["elementId(source_candidate)"],
                    "destination_name": destination,
                    "destination_embedding": dest_embedding,
                    "user_id": user_id,
                    "source_node_props": source_node_props,
                    "dest_node_props": dest_node_props,
                    "edge_props": edge_props,
                }
                if agent_id:
                    params["agent_id"] = agent_id
                if run_id:
                    params["run_id"] = run_id

            elif destination_node_search_result and not source_node_search_result:
                # Build source MERGE properties
                merge_props = ["name: $source_name", "user_id: $user_id"]
                if agent_id:
                    merge_props.append("agent_id: $agent_id")
                if run_id:
                    merge_props.append("run_id: $run_id")
                merge_props_str = ", ".join(merge_props)

                cypher = f"""
                MATCH (destination)
                WHERE elementId(destination) = $destination_id
                SET destination.mentions = coalesce(destination.mentions, 0) + 1
                SET destination += $dest_node_props
                WITH destination
                MERGE (source {source_label} {{{merge_props_str}}})
                ON CREATE SET
                    source.created = timestamp(),
                    source.mentions = 1
                    {source_extra_set}
                ON MATCH SET
                    source.mentions = coalesce(source.mentions, 0) + 1
                WITH source, destination
                SET source += $source_node_props
                WITH source, destination
                CALL db.create.setNodeVectorProperty(source, 'embedding', $source_embedding)
                WITH source, destination
                MERGE (source)-[r:{relationship}]->(destination)
                ON CREATE SET
                    r.created_at = timestamp(),
                    r.updated_at = timestamp(),
                    r.mentions = 1,
                    r.valid = true
                    {edge_props_set}
                ON MATCH SET
                    r.mentions = coalesce(r.mentions, 0) + 1,
                    r.valid = true,
                    r.updated_at = timestamp(),
                    r.invalidated_at = null
                    {edge_props_set}
                RETURN source.name AS source, type(r) AS relationship, destination.name AS target
                """

                params = {
                    "destination_id": destination_node_search_result[0]["elementId(destination_candidate)"],
                    "source_name": source,
                    "source_embedding": source_embedding,
                    "user_id": user_id,
                    "source_node_props": source_node_props,
                    "dest_node_props": dest_node_props,
                    "edge_props": edge_props,
                }
                if agent_id:
                    params["agent_id"] = agent_id
                if run_id:
                    params["run_id"] = run_id

            elif source_node_search_result and destination_node_search_result:
                cypher = f"""
                MATCH (source)
                WHERE elementId(source) = $source_id
                SET source.mentions = coalesce(source.mentions, 0) + 1
                SET source += $source_node_props
                WITH source
                MATCH (destination)
                WHERE elementId(destination) = $destination_id
                SET destination.mentions = coalesce(destination.mentions, 0) + 1
                SET destination += $dest_node_props
                MERGE (source)-[r:{relationship}]->(destination)
                ON CREATE SET
                    r.created_at = timestamp(),
                    r.updated_at = timestamp(),
                    r.mentions = 1,
                    r.valid = true
                    {edge_props_set}
                ON MATCH SET
                    r.mentions = coalesce(r.mentions, 0) + 1,
                    r.valid = true,
                    r.updated_at = timestamp(),
                    r.invalidated_at = null
                    {edge_props_set}
                RETURN source.name AS source, type(r) AS relationship, destination.name AS target
                """

                params = {
                    "source_id": source_node_search_result[0]["elementId(source_candidate)"],
                    "destination_id": destination_node_search_result[0]["elementId(destination_candidate)"],
                    "user_id": user_id,
                    "source_node_props": source_node_props,
                    "dest_node_props": dest_node_props,
                    "edge_props": edge_props,
                }
                if agent_id:
                    params["agent_id"] = agent_id
                if run_id:
                    params["run_id"] = run_id

            else:
                # Build dynamic MERGE props for both source and destination
                source_merge = ["name: $source_name", "user_id: $user_id"]
                dest_merge = ["name: $dest_name", "user_id: $user_id"]
                if agent_id:
                    source_merge.append("agent_id: $agent_id")
                    dest_merge.append("agent_id: $agent_id")
                if run_id:
                    source_merge.append("run_id: $run_id")
                    dest_merge.append("run_id: $run_id")
                source_merge_str = ", ".join(source_merge)
                dest_merge_str = ", ".join(dest_merge)

                cypher = f"""
                MERGE (source {source_label} {{{source_merge_str}}})
                ON CREATE SET source.created = timestamp(),
                            source.mentions = 1
                            {source_extra_set}
                ON MATCH SET source.mentions = coalesce(source.mentions, 0) + 1
                WITH source
                SET source += $source_node_props
                WITH source
                CALL db.create.setNodeVectorProperty(source, 'embedding', $source_embedding)
                WITH source
                MERGE (destination {destination_label} {{{dest_merge_str}}})
                ON CREATE SET destination.created = timestamp(),
                            destination.mentions = 1
                            {destination_extra_set}
                ON MATCH SET destination.mentions = coalesce(destination.mentions, 0) + 1
                WITH source, destination
                SET destination += $dest_node_props
                WITH source, destination
                CALL db.create.setNodeVectorProperty(destination, 'embedding', $dest_embedding)
                WITH source, destination
                MERGE (source)-[r:{relationship}]->(destination)
                ON CREATE SET
                    r.created_at = timestamp(),
                    r.updated_at = timestamp(),
                    r.mentions = 1,
                    r.valid = true
                    {edge_props_set}
                ON MATCH SET
                    r.mentions = coalesce(r.mentions, 0) + 1,
                    r.valid = true,
                    r.updated_at = timestamp(),
                    r.invalidated_at = null
                    {edge_props_set}
                RETURN source.name AS source, type(r) AS relationship, destination.name AS target
                """

                params = {
                    "source_name": source,
                    "dest_name": destination,
                    "source_embedding": source_embedding,
                    "dest_embedding": dest_embedding,
                    "user_id": user_id,
                    "source_node_props": source_node_props,
                    "dest_node_props": dest_node_props,
                    "edge_props": edge_props,
                }
                if agent_id:
                    params["agent_id"] = agent_id
                if run_id:
                    params["run_id"] = run_id
            result = self.graph.query(cypher, params=params)
            results.append(result)
        return results

    # ------------------------------------------------------------------
    # Phase 2 — "Me" anchor node helpers
    # ------------------------------------------------------------------

    def _ensure_me_node(self, filters):
        """Create the anchor "Me" node for the current scope if it doesn't exist.

        Uses MERGE so the node is created only once per (user_id[, agent_id[, run_id]]).
        The node is tagged with ``is_anchor = true`` and ``entity_type = "self"``.
        """
        anchor_name = self.anchor_node_name

        merge_props = ["name: $anchor_name", "user_id: $user_id"]
        params = {"anchor_name": anchor_name, "user_id": filters["user_id"]}
        if filters.get("agent_id"):
            merge_props.append("agent_id: $agent_id")
            params["agent_id"] = filters["agent_id"]
        if filters.get("run_id"):
            merge_props.append("run_id: $run_id")
            params["run_id"] = filters["run_id"]
        merge_props_str = ", ".join(merge_props)

        # Embed the anchor name so vector-similarity searches can find it
        anchor_embedding = self.embedding_model.embed(anchor_name)
        params["anchor_embedding"] = anchor_embedding

        cypher = f"""
        MERGE (me {self.node_label} {{{merge_props_str}}})
        ON CREATE SET
            me.created = timestamp(),
            me.created_at = timestamp(),
            me.updated_at = timestamp(),
            me.entity_type = 'self',
            me.is_anchor = true,
            me.mentions = 1
        ON MATCH SET
            me.mentions = coalesce(me.mentions, 0) + 1,
            me.updated_at = timestamp()
        WITH me
        CALL db.create.setNodeVectorProperty(me, 'embedding', $anchor_embedding)
        RETURN me.name AS name
        """
        self.graph.query(cypher, params=params)

    def _connect_orphans_to_me(self, to_be_added, filters):
        """Connect newly-added nodes that have no other connections to the anchor "Me" node.

        An "orphan" is a node whose *only* valid relationships (if any) are to/from
        the "Me" node itself.  For each unique node name in ``to_be_added`` that is
        not already connected to another non-anchor node, a ``KNOWS_ABOUT``
        relationship is created from "Me" → orphan.
        """
        if not to_be_added:
            return

        anchor_name = self.anchor_node_name

        # Collect every node name from the entities that were just added
        node_names = set()
        for item in to_be_added:
            src = item.get("source", "").lower().replace(" ", "_")
            dst = item.get("destination", "").lower().replace(" ", "_")
            if src:
                node_names.add(src)
            if dst:
                node_names.add(dst)

        # Never try to connect the anchor to itself
        node_names.discard(anchor_name)
        if not node_names:
            return

        # Build scope filter
        node_filter_parts = ["n.user_id = $user_id"]
        params = {
            "user_id": filters["user_id"],
            "anchor_name": anchor_name,
            "node_names": list(node_names),
        }
        if filters.get("agent_id"):
            node_filter_parts.append("n.agent_id = $agent_id")
            params["agent_id"] = filters["agent_id"]
        if filters.get("run_id"):
            node_filter_parts.append("n.run_id = $run_id")
            params["run_id"] = filters["run_id"]
        node_filter_str = " AND ".join(node_filter_parts)

        # Build anchor scope filter (same scope conditions but for the "me" node)
        me_filter_parts = ["me.user_id = $user_id"]
        if filters.get("agent_id"):
            me_filter_parts.append("me.agent_id = $agent_id")
        if filters.get("run_id"):
            me_filter_parts.append("me.run_id = $run_id")
        me_filter_str = " AND ".join(me_filter_parts)

        # Find orphan nodes: nodes that have no valid relationship to any
        # non-anchor node.  Then connect them to "Me" with KNOWS_ABOUT.
        cypher = f"""
        MATCH (n {self.node_label})
        WHERE {node_filter_str} AND n.name IN $node_names AND n.is_anchor IS NULL
        WITH n
        OPTIONAL MATCH (n)-[r]-(other {self.node_label})
        WHERE other.is_anchor IS NULL AND (r.valid IS NULL OR r.valid = true)
        WITH n, count(other) AS connections
        WHERE connections = 0
        WITH collect(n) AS orphans
        MATCH (me {self.node_label})
        WHERE me.name = $anchor_name AND {me_filter_str}
        UNWIND orphans AS orphan
        MERGE (me)-[r:KNOWS_ABOUT]->(orphan)
        ON CREATE SET
            r.created_at = timestamp(),
            r.updated_at = timestamp(),
            r.mentions = 1,
            r.valid = true
        ON MATCH SET
            r.mentions = coalesce(r.mentions, 0) + 1,
            r.valid = true,
            r.updated_at = timestamp(),
            r.invalidated_at = null
        RETURN orphan.name AS orphan_name
        """

        result = self.graph.query(cypher, params=params)
        if result:
            orphan_names = [r["orphan_name"] for r in result]
            logger.debug(f"Connected orphan nodes to '{anchor_name}': {orphan_names}")

    def get_me_node(self, filters, depth=1):
        """Return the anchor "Me" node and its connections up to ``depth`` hops.

        Args:
            filters (dict): Scope filters (user_id, agent_id, run_id).
            depth (int): How many hops out from the "Me" node to traverse.
                         Defaults to 1 (direct connections only).

        Returns:
            dict: ``{"me": {...}, "connections": [...]}``.  Each connection
            contains ``source``, ``relationship``, ``destination`` and optional
            ``edge_properties``, ``destination_properties`` dicts.
            Returns ``None`` if no "Me" node exists for the scope.
        """
        anchor_name = self.anchor_node_name
        depth = max(1, int(depth))  # Sanitize: must be positive integer

        # Build match conditions for the anchor node
        me_props = ["name: $anchor_name", "user_id: $user_id"]
        params = {"anchor_name": anchor_name, "user_id": filters["user_id"]}
        if filters.get("agent_id"):
            me_props.append("agent_id: $agent_id")
            params["agent_id"] = filters["agent_id"]
        if filters.get("run_id"):
            me_props.append("run_id: $run_id")
            params["run_id"] = filters["run_id"]
        me_props_str = ", ".join(me_props)

        # Variable-length path: 1..depth hops.
        # Neo4j does not support parameterized bounds in variable-length paths,
        # so we interpolate depth (validated int) directly into the Cypher string.
        # r is always a list of relationships from the *1..N pattern.
        cypher = f"""
        MATCH (me {self.node_label} {{{me_props_str}}})
        OPTIONAL MATCH path = (me)-[*1..{depth}]->(neighbor {self.node_label})
        WHERE ALL(rel IN relationships(path) WHERE rel.valid IS NULL OR rel.valid = true)
        WITH me, relationships(path) AS rels, neighbor
        RETURN me {{.name, .entity_type, .is_anchor, .created}} AS me_node,
               CASE WHEN neighbor IS NOT NULL THEN {{
                   source: me.name,
                   relationship: type(last(rels)),
                   destination: neighbor.name,
                   edge_properties: properties(last(rels)),
                   destination_properties: properties(neighbor)
               }} ELSE null END AS connection
        """

        results = self.graph.query(cypher, params=params)

        if not results:
            return None

        me_data = results[0].get("me_node")
        connections = []
        for row in results:
            conn = row.get("connection")
            if conn is not None:
                # Filter system keys from properties
                edge_props = {k: v for k, v in (conn.get("edge_properties") or {}).items() if k not in _SYSTEM_RESERVED_KEYS}
                dest_props = {k: v for k, v in (conn.get("destination_properties") or {}).items() if k not in _SYSTEM_RESERVED_KEYS}
                entry = {
                    "source": conn["source"],
                    "relationship": conn["relationship"],
                    "destination": conn["destination"],
                }
                if edge_props:
                    entry["edge_properties"] = edge_props
                if dest_props:
                    entry["destination_properties"] = dest_props
                connections.append(entry)

        return {"me": dict(me_data) if me_data else {}, "connections": connections}

    # ------------------------------------------------------------------
    # Phase 3 — Multi-hop graph walking
    # ------------------------------------------------------------------

    def _build_scope_filter(self, alias, filters, *, include_name=None):
        """Build a Cypher WHERE clause and params dict for scope filtering.

        Args:
            alias (str): The Cypher variable alias (e.g. ``"n"``).
            filters (dict): Scope filters (user_id, agent_id, run_id).
            include_name (str | None): If given, add a ``name`` equality check.

        Returns:
            tuple[str, dict]: ``(where_clause, params)``
        """
        parts = [f"{alias}.user_id = $user_id"]
        params = {"user_id": filters["user_id"]}
        if filters.get("agent_id"):
            parts.append(f"{alias}.agent_id = $agent_id")
            params["agent_id"] = filters["agent_id"]
        if filters.get("run_id"):
            parts.append(f"{alias}.run_id = $run_id")
            params["run_id"] = filters["run_id"]
        if include_name is not None:
            parts.append(f"{alias}.name = $node_name")
            params["node_name"] = include_name
        return " AND ".join(parts), params

    def get_node(self, node_name, filters):
        """Retrieve a specific node by name with all its properties and edge count.

        Args:
            node_name (str): Name of the node to retrieve.
            filters (dict): Scope filters (user_id, agent_id, run_id).

        Returns:
            dict | None: Node info with ``name``, ``entity_type``, ``properties``,
            ``edge_count`` and ``is_anchor`` flag.  ``None`` if not found.
        """
        where_clause, params = self._build_scope_filter("n", filters, include_name=node_name)

        cypher = f"""
        MATCH (n {self.node_label})
        WHERE {where_clause}
        OPTIONAL MATCH (n)-[r]-()
        WHERE r.valid IS NULL OR r.valid = true
        RETURN n AS node, properties(n) AS props, count(r) AS edge_count
        """

        results = self.graph.query(cypher, params=params)
        if not results:
            return None

        row = results[0]
        all_props = row.get("props") or {}
        user_props = {k: v for k, v in all_props.items() if k not in _SYSTEM_RESERVED_KEYS}

        return {
            "name": all_props.get("name", node_name),
            "entity_type": all_props.get("entity_type"),
            "is_anchor": bool(all_props.get("is_anchor")),
            "properties": user_props,
            "edge_count": row.get("edge_count", 0),
        }

    def get_neighbors(self, node_name, filters, direction="both",
                      relationship_type=None, limit=100):
        """Get all nodes directly connected to a given node (1-hop).

        Args:
            node_name (str): Starting node name.
            filters (dict): Scope filters.
            direction (str): ``"outgoing"``, ``"incoming"`` or ``"both"`` (default).
            relationship_type (str | None): Optional relationship type filter.
            limit (int): Maximum neighbours to return.

        Returns:
            list[dict]: Each entry has ``source``, ``relationship``,
            ``destination``, and optional ``edge_properties`` /
            ``destination_properties`` dicts.
        """
        where_clause, params = self._build_scope_filter("n", filters, include_name=node_name)
        params["limit"] = limit

        # Build relationship type filter
        rel_type_fragment = f":{relationship_type}" if relationship_type else ""

        # Direction-specific match patterns
        if direction == "outgoing":
            match = f"(n)-[r{rel_type_fragment}]->(neighbor {self.node_label})"
            return_clause = "n.name AS source, type(r) AS relationship, neighbor.name AS destination"
        elif direction == "incoming":
            match = f"(n)<-[r{rel_type_fragment}]-(neighbor {self.node_label})"
            return_clause = "neighbor.name AS source, type(r) AS relationship, n.name AS destination"
        else:  # both
            match = f"(n)-[r{rel_type_fragment}]-(neighbor {self.node_label})"
            # For undirected match, determine direction from startNode
            return_clause = (
                "CASE WHEN startNode(r) = n THEN n.name ELSE neighbor.name END AS source, "
                "type(r) AS relationship, "
                "CASE WHEN startNode(r) = n THEN neighbor.name ELSE n.name END AS destination"
            )

        cypher = f"""
        MATCH (n {self.node_label})
        WHERE {where_clause}
        MATCH {match}
        WHERE (r.valid IS NULL OR r.valid = true)
        RETURN {return_clause},
               properties(r) AS edge_properties,
               properties(neighbor) AS neighbor_properties
        LIMIT $limit
        """

        results = self.graph.query(cypher, params=params)

        neighbors = []
        for row in results:
            entry = {
                "source": row["source"],
                "relationship": row["relationship"],
                "destination": row["destination"],
            }
            edge_props = {k: v for k, v in (row.get("edge_properties") or {}).items() if k not in _SYSTEM_RESERVED_KEYS}
            dest_props = {k: v for k, v in (row.get("neighbor_properties") or {}).items() if k not in _SYSTEM_RESERVED_KEYS}
            if edge_props:
                entry["edge_properties"] = edge_props
            if dest_props:
                entry["destination_properties"] = dest_props
            neighbors.append(entry)

        return neighbors

    def walk(self, start_node, filters, depth=2, relationship_type=None, limit=100):
        """Walk the graph from a starting node up to N hops.

        Uses Cypher variable-length paths.  Only follows edges where
        ``valid`` is null or true (soft-deleted edges are skipped).

        Args:
            start_node (str): Name of the starting node.
            filters (dict): Scope filters.
            depth (int): Maximum hops (1-5, default 2).
            relationship_type (str | None): Optional relationship type filter.
            limit (int): Maximum result rows.

        Returns:
            list[dict]: Each entry has ``source``, ``relationship``,
            ``destination``, optional ``edge_properties`` /
            ``destination_properties``, and ``depth`` (hop count from start).
        """
        depth = max(1, min(5, int(depth)))  # Clamp 1..5
        where_clause, params = self._build_scope_filter("start", filters, include_name=start_node)
        params["limit"] = limit

        rel_type_fragment = f":{relationship_type}" if relationship_type else ""

        # Neo4j doesn't support parameterized variable-length bounds,
        # so we interpolate the validated int directly.
        cypher = f"""
        MATCH (start {self.node_label})
        WHERE {where_clause}
        MATCH path = (start)-[{rel_type_fragment}*1..{depth}]-(end {self.node_label})
        WHERE ALL(rel IN relationships(path) WHERE rel.valid IS NULL OR rel.valid = true)
        WITH path, end, relationships(path) AS rels, length(path) AS hop_depth
        UNWIND range(0, size(rels) - 1) AS idx
        WITH rels[idx] AS r, nodes(path)[idx] AS src, nodes(path)[idx + 1] AS dst, hop_depth
        RETURN DISTINCT
            src.name AS source,
            type(r) AS relationship,
            dst.name AS destination,
            properties(r) AS edge_properties,
            properties(dst) AS destination_properties,
            hop_depth AS depth
        LIMIT $limit
        """

        results = self.graph.query(cypher, params=params)

        walked = []
        for row in results:
            entry = {
                "source": row["source"],
                "relationship": row["relationship"],
                "destination": row["destination"],
                "depth": row.get("depth", 1),
            }
            edge_props = {k: v for k, v in (row.get("edge_properties") or {}).items() if k not in _SYSTEM_RESERVED_KEYS}
            dest_props = {k: v for k, v in (row.get("destination_properties") or {}).items() if k not in _SYSTEM_RESERVED_KEYS}
            if edge_props:
                entry["edge_properties"] = edge_props
            if dest_props:
                entry["destination_properties"] = dest_props
            walked.append(entry)

        return walked

    def find_path(self, from_node, to_node, filters, max_depth=5):
        """Find the shortest path between two nodes.

        Uses Cypher's ``shortestPath()`` function.  Only traverses edges
        where ``valid`` is null or true.

        Args:
            from_node (str): Source node name.
            to_node (str): Target node name.
            filters (dict): Scope filters.
            max_depth (int): Maximum path length (default 5).

        Returns:
            list[dict] | None: Ordered list of hops from source to target,
            each with ``source``, ``relationship``, ``destination`` and
            optional property dicts.  ``None`` if no path exists.
        """
        max_depth = max(1, min(10, int(max_depth)))  # Clamp 1..10
        from_where, params = self._build_scope_filter("a", filters, include_name=from_node)

        # Add to_node param with a distinct name
        to_parts = ["b.user_id = $user_id"]
        params["to_node_name"] = to_node
        if filters.get("agent_id"):
            to_parts.append("b.agent_id = $agent_id")
        if filters.get("run_id"):
            to_parts.append("b.run_id = $run_id")
        to_parts.append("b.name = $to_node_name")
        to_where = " AND ".join(to_parts)

        cypher = f"""
        MATCH (a {self.node_label}), (b {self.node_label})
        WHERE {from_where} AND {to_where}
        MATCH path = shortestPath((a)-[*1..{max_depth}]-(b))
        WHERE ALL(rel IN relationships(path) WHERE rel.valid IS NULL OR rel.valid = true)
        RETURN relationships(path) AS rels, nodes(path) AS path_nodes
        """

        results = self.graph.query(cypher, params=params)
        if not results:
            return None

        row = results[0]
        rels = row.get("rels", [])
        path_nodes = row.get("path_nodes", [])

        hops = []
        for i, rel in enumerate(rels):
            rel_props = dict(rel) if isinstance(rel, dict) else (dict(rel) if hasattr(rel, 'items') else {})
            src_node = path_nodes[i] if i < len(path_nodes) else {}
            dst_node = path_nodes[i + 1] if (i + 1) < len(path_nodes) else {}

            # Extract names — handle both dict and Neo4j node objects
            src_name = src_node.get("name", "") if isinstance(src_node, dict) else getattr(src_node, "name", str(src_node))
            dst_name = dst_node.get("name", "") if isinstance(dst_node, dict) else getattr(dst_node, "name", str(dst_node))

            # Relationship type — handle both str and Neo4j relationship objects
            rel_type = rel_props.pop("type", None) or (type(rel).__name__ if not isinstance(rel, dict) else "")
            # If it's a Neo4j relationship, try to get the type
            if hasattr(rel, "type"):
                rel_type = rel.type

            edge_props = {k: v for k, v in rel_props.items() if k not in _SYSTEM_RESERVED_KEYS}
            dst_props_raw = dict(dst_node) if isinstance(dst_node, dict) else {}
            dst_props = {k: v for k, v in dst_props_raw.items() if k not in _SYSTEM_RESERVED_KEYS}

            entry = {
                "source": src_name,
                "relationship": rel_type,
                "destination": dst_name,
            }
            if edge_props:
                entry["edge_properties"] = edge_props
            if dst_props:
                entry["destination_properties"] = dst_props
            hops.append(entry)

        return hops

    def get_edges(self, node_name, filters, direction="both",
                  relationship_type=None, include_invalid=False, limit=100):
        """Get all edges for a node with optional filtering.

        Args:
            node_name (str): Node name.
            filters (dict): Scope filters.
            direction (str): ``"outgoing"``, ``"incoming"`` or ``"both"`` (default).
            relationship_type (str | None): Optional relationship type filter.
            include_invalid (bool): If True, also return soft-deleted edges.
            limit (int): Maximum edges to return.

        Returns:
            list[dict]: Each entry has ``source``, ``relationship``,
            ``destination``, ``valid``, and optional ``edge_properties`` dict.
        """
        where_clause, params = self._build_scope_filter("n", filters, include_name=node_name)
        params["limit"] = limit

        rel_type_fragment = f":{relationship_type}" if relationship_type else ""
        valid_filter = "" if include_invalid else "AND (r.valid IS NULL OR r.valid = true)"

        if direction == "outgoing":
            match = f"(n)-[r{rel_type_fragment}]->(other {self.node_label})"
            return_clause = "n.name AS source, type(r) AS relationship, other.name AS destination"
        elif direction == "incoming":
            match = f"(n)<-[r{rel_type_fragment}]-(other {self.node_label})"
            return_clause = "other.name AS source, type(r) AS relationship, n.name AS destination"
        else:
            match = f"(n)-[r{rel_type_fragment}]-(other {self.node_label})"
            return_clause = (
                "CASE WHEN startNode(r) = n THEN n.name ELSE other.name END AS source, "
                "type(r) AS relationship, "
                "CASE WHEN startNode(r) = n THEN other.name ELSE n.name END AS destination"
            )

        cypher = f"""
        MATCH (n {self.node_label})
        WHERE {where_clause}
        MATCH {match}
        WHERE true {valid_filter}
        RETURN {return_clause},
               properties(r) AS edge_properties,
               r.valid AS valid
        LIMIT $limit
        """

        results = self.graph.query(cypher, params=params)

        edges = []
        for row in results:
            entry = {
                "source": row["source"],
                "relationship": row["relationship"],
                "destination": row["destination"],
                "valid": row.get("valid") is not False,  # None → True (legacy)
            }
            edge_props = {k: v for k, v in (row.get("edge_properties") or {}).items() if k not in _SYSTEM_RESERVED_KEYS}
            if edge_props:
                entry["edge_properties"] = edge_props
            edges.append(entry)

        return edges

    def _remove_spaces_from_entities(self, entity_list):
        return remove_spaces_from_entities(entity_list, sanitize_relationship=True)

    def _search_source_node(self, source_embedding, filters, threshold=0.9):
        # Build WHERE conditions
        where_conditions = ["source_candidate.embedding IS NOT NULL", "source_candidate.user_id = $user_id"]
        if filters.get("agent_id"):
            where_conditions.append("source_candidate.agent_id = $agent_id")
        if filters.get("run_id"):
            where_conditions.append("source_candidate.run_id = $run_id")
        where_clause = " AND ".join(where_conditions)

        cypher = f"""
            MATCH (source_candidate {self.node_label})
            WHERE {where_clause}

            WITH source_candidate,
            round(2 * vector.similarity.cosine(source_candidate.embedding, $source_embedding) - 1, 4) AS source_similarity // denormalize for backward compatibility
            WHERE source_similarity >= $threshold

            WITH source_candidate, source_similarity
            ORDER BY source_similarity DESC
            LIMIT 1

            RETURN elementId(source_candidate)
            """

        params = {
            "source_embedding": source_embedding,
            "user_id": filters["user_id"],
            "threshold": threshold,
        }
        if filters.get("agent_id"):
            params["agent_id"] = filters["agent_id"]
        if filters.get("run_id"):
            params["run_id"] = filters["run_id"]

        result = self.graph.query(cypher, params=params)
        return result

    def _search_destination_node(self, destination_embedding, filters, threshold=0.9):
        # Build WHERE conditions
        where_conditions = ["destination_candidate.embedding IS NOT NULL", "destination_candidate.user_id = $user_id"]
        if filters.get("agent_id"):
            where_conditions.append("destination_candidate.agent_id = $agent_id")
        if filters.get("run_id"):
            where_conditions.append("destination_candidate.run_id = $run_id")
        where_clause = " AND ".join(where_conditions)

        cypher = f"""
            MATCH (destination_candidate {self.node_label})
            WHERE {where_clause}

            WITH destination_candidate,
            round(2 * vector.similarity.cosine(destination_candidate.embedding, $destination_embedding) - 1, 4) AS destination_similarity // denormalize for backward compatibility

            WHERE destination_similarity >= $threshold

            WITH destination_candidate, destination_similarity
            ORDER BY destination_similarity DESC
            LIMIT 1

            RETURN elementId(destination_candidate)
            """

        params = {
            "destination_embedding": destination_embedding,
            "user_id": filters["user_id"],
            "threshold": threshold,
        }
        if filters.get("agent_id"):
            params["agent_id"] = filters["agent_id"]
        if filters.get("run_id"):
            params["run_id"] = filters["run_id"]

        result = self.graph.query(cypher, params=params)
        return result

    # Reset is not defined in base.py
    def reset(self):
        """Reset the graph by clearing all nodes and relationships."""
        logger.warning("Clearing graph...")
        cypher_query = """
        MATCH (n) DETACH DELETE n
        """
        return self.graph.query(cypher_query)
