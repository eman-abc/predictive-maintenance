"""Unity Catalog discovery via Databricks REST (works when MlflowClient has no search_catalogs)."""

from __future__ import annotations

import os
from typing import Any

import requests

_SKIP_CATALOG_PREFIXES = ("__", "system")


def _headers(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def list_uc_catalogs(host: str, token: str) -> list[str]:
    url = f"{host.rstrip('/')}/api/2.1/unity-catalog/catalogs"
    resp = requests.get(url, headers=_headers(token), timeout=60)
    if resp.status_code == 403:
        raise PermissionError(
            "403 on Unity Catalog API — your PAT cannot list catalogs. "
            "In Databricks UI: Catalog Explorer → note catalog + schema names, then set "
            "MLFLOW_UC_CATALOG and MLFLOW_UC_SCHEMA. Or set MLFLOW_REGISTER_MODELS=log_only "
            "to skip UC registry (experiments still work)."
        )
    resp.raise_for_status()
    data = resp.json()
    names = [c["name"] for c in data.get("catalogs", []) if c.get("name")]
    return sorted(names)


def list_uc_schemas(host: str, token: str, catalog: str) -> list[str]:
    url = f"{host.rstrip('/')}/api/2.1/unity-catalog/schemas"
    resp = requests.get(
        url,
        headers=_headers(token),
        params={"catalog_name": catalog},
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()
    names = [s["name"] for s in data.get("schemas", []) if s.get("name")]
    return sorted(names)


def pick_uc_catalog(catalogs: list[str], preferred: str | None = None) -> str | None:
    if preferred and preferred in catalogs:
        return preferred
    for name in catalogs:
        if any(name.startswith(p) for p in _SKIP_CATALOG_PREFIXES):
            continue
        return name
    return catalogs[0] if catalogs else None


def pick_uc_schema(schemas: list[str], preferred: str | None = None) -> str | None:
    if preferred and preferred in schemas:
        return preferred
    for name in schemas:
        if name == "default":
            return name
        if name == "information_schema":
            continue
        return name
    return schemas[0] if schemas else None


def resolve_uc_catalog_schema(
    host: str | None = None,
    token: str | None = None,
    *,
    catalog: str | None = None,
    schema: str | None = None,
) -> tuple[str, str, dict[str, Any]]:
    """
    Resolve catalog + schema for UC model registration.

    Uses MLFLOW_UC_* env vars, then REST discovery. Raises if nothing usable.
    """
    host = host or os.getenv("DATABRICKS_HOST", "")
    token = token or os.getenv("DATABRICKS_TOKEN", "")
    if not host or not token:
        raise ValueError("DATABRICKS_HOST and DATABRICKS_TOKEN required")

    catalog = (catalog or os.getenv("MLFLOW_UC_CATALOG", "")).strip() or None
    schema = (schema or os.getenv("MLFLOW_UC_SCHEMA", "")).strip() or None

    catalogs = list_uc_catalogs(host, token)
    if not catalogs:
        raise RuntimeError(
            "No Unity Catalogs visible to this token. "
            "In Databricks: Catalog Explorer → create a catalog, or widen PAT permissions."
        )

    chosen_cat = pick_uc_catalog(catalogs, catalog)
    if not chosen_cat:
        raise RuntimeError("Could not pick a catalog.")

    schemas = list_uc_schemas(host, token, chosen_cat)
    if not schemas:
        raise RuntimeError(
            f"No schemas in catalog {chosen_cat!r}. Create one in Catalog Explorer "
            f"(e.g. schema 'cmapss')."
        )

    chosen_schema = pick_uc_schema(schemas, schema)
    if not chosen_schema:
        raise RuntimeError(f"Could not pick a schema in {chosen_cat}.")

    meta = {
        "catalogs_available": catalogs,
        "schemas_available": schemas,
        "catalog_requested": catalog,
        "schema_requested": schema,
        "catalog_chosen": chosen_cat,
        "schema_chosen": chosen_schema,
    }
    return chosen_cat, chosen_schema, meta
