# Phase 7: Storage Providers

> **Before starting:** Look up what has been implemented in the previous phases. Inspect the actual codebase — don't assume anything was implemented as originally planned.

## Goal

Support **`storage://` URLs** for loading documents from cloud storage providers (OneDrive/SharePoint, Google Drive, Dropbox, Nextcloud). When the system encounters a `storage://driveId/fileId` URL, it fetches the document content, vectorizes it, and creates a Document node in the graph.

## Current State (Inspect Before Starting)

Check these files:
- `mem0/memory/graph_memory.py` — Document node support from Phase 5
- `mem0/memory/main.py` — `graph_add_document()`
- Any existing URL handling in the codebase

## Changes Required

### 1. Storage Provider Interface

Create `mem0/storage_providers/base.py`:

```python
from abc import ABC, abstractmethod

class StorageProvider(ABC):
    """Abstract base class for cloud storage providers."""
    
    @abstractmethod
    def fetch_document(self, file_id, drive_id=None):
        """Fetch document content from storage.
        
        Args:
            file_id: The file identifier in the storage system
            drive_id: Optional drive/site identifier
            
        Returns:
            dict: {
                "content": "document text...",
                "title": "filename.docx",
                "mime_type": "application/vnd.openxmlformats-officedocument...",
                "metadata": {"author": "...", "modified": "..."}
            }
        """
        pass
    
    @abstractmethod
    def supports_url(self, url):
        """Check if this provider can handle the given URL."""
        pass
```

### 2. URL Parser

Create `mem0/storage_providers/url_parser.py`:

```python
def parse_storage_url(url):
    """Parse a storage:// URL into components.
    
    Formats:
        storage://fileId           → default drive, file by ID
        storage://driveId/fileId   → specific drive, file by ID
        
    Returns:
        dict: {"drive_id": str|None, "file_id": str}
    """
```

### 3. Provider Implementations

**OneDrive/SharePoint** — `mem0/storage_providers/onedrive.py`:
- Uses Microsoft Graph API
- Config: `tenant_id`, `client_id`, `client_secret` or managed identity
- Supports: `storage://siteId/fileId` or `storage://fileId`

**Google Drive** — `mem0/storage_providers/google_drive.py`:
- Uses Google Drive API v3
- Config: `service_account_json` or `credentials_path`
- Supports: `storage://driveId/fileId` or `storage://fileId`

**Dropbox** — `mem0/storage_providers/dropbox.py`:
- Uses Dropbox API
- Config: `access_token` or `app_key`/`app_secret`

**Nextcloud** — `mem0/storage_providers/nextcloud.py`:
- Uses WebDAV/Nextcloud API
- Config: `url`, `username`, `password`

### 4. Storage Provider Factory

Create `mem0/storage_providers/factory.py`:

```python
class StorageProviderFactory:
    providers = {
        "onedrive": "mem0.storage_providers.onedrive.OneDriveProvider",
        "sharepoint": "mem0.storage_providers.onedrive.OneDriveProvider",
        "google_drive": "mem0.storage_providers.google_drive.GoogleDriveProvider",
        "dropbox": "mem0.storage_providers.dropbox.DropboxProvider",
        "nextcloud": "mem0.storage_providers.nextcloud.NextcloudProvider",
    }
    
    @classmethod
    def create(cls, provider_name, config):
        ...
```

### 5. Configuration

Add to `mem0/configs/base.py` or a new `mem0/configs/storage.py`:

```python
class StorageConfig(BaseModel):
    default_provider: Optional[str] = None  # e.g., "onedrive"
    providers: Dict[str, Dict] = {}  # provider configs by name
```

Add to `MemoryConfig`:
```python
storage: Optional[StorageConfig] = None
```

### 6. Integration with Document Nodes

Modify `Memory.graph_add_document()` in main.py:
- If `source_url` starts with `storage://`, parse it, fetch via storage provider, then proceed
- If `source_url` is a regular URL, use it as metadata only (content must be provided)

### 7. Content Extraction

For binary documents (Word, PDF, etc.), add simple extraction:
- Use `python-docx` for .docx
- Use `PyPDF2` or `pdfplumber` for .pdf
- Use `openpyxl` for .xlsx
- Plain text for .txt, .md, .csv

Create `mem0/storage_providers/extractors.py` with a `extract_text(content_bytes, mime_type)` function.

## Testing

- Test URL parsing for all formats
- Test each provider with mocked API calls
- Test content extraction for each supported file type
- Test end-to-end: `storage://` URL → fetch → chunk → vectorize → graph node
- Test error handling for missing providers, auth failures, unsupported file types

## Files Created

- `mem0/storage_providers/__init__.py`
- `mem0/storage_providers/base.py`
- `mem0/storage_providers/url_parser.py`
- `mem0/storage_providers/factory.py`
- `mem0/storage_providers/onedrive.py`
- `mem0/storage_providers/google_drive.py`
- `mem0/storage_providers/dropbox.py`
- `mem0/storage_providers/nextcloud.py`
- `mem0/storage_providers/extractors.py`
- `mem0/configs/storage.py`
- `tests/storage_providers/test_url_parser.py` (new)
- `tests/storage_providers/test_providers.py` (new)

## Files Modified

- `mem0/configs/base.py`
- `mem0/memory/main.py`
- `pyproject.toml` (new optional dependencies)
