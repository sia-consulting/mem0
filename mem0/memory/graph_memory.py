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
    "invalidated_at", "relationship",
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

    def add(self, data, filters):
        """
        Adds data to the graph.

        Args:
            data (str): The data to add to the graph.
            filters (dict): A dictionary containing filters to be applied during the addition.
        """
        entity_type_map = self._retrieve_nodes_from_data(data, filters)
        to_be_added = self._establish_nodes_relations_from_data(data, filters, entity_type_map)
        search_output = self._search_graph_db(node_list=list(entity_type_map.keys()), filters=filters)
        to_be_deleted = self._get_delete_entities_from_search_output(search_output, data, filters)

        # TODO: Batch queries with APOC plugin
        # TODO: Add more filter support
        deleted_entities = self._delete_entities(to_be_deleted, filters)
        added_entities = self._add_entities(to_be_added, filters, entity_type_map)

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
