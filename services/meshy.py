"""
Meshy AI text-to-3D client.

Flow:
  task_id = create_task(prompt, ...)
  result  = poll_task(task_id)          # blocks until SUCCEEDED or FAILED
  paths   = download_model(result, dir) # saves GLB + OBJ locally
"""

import time
import httpx
from pathlib import Path
from typing import Optional

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from agent.config import (
    MESHY_API_KEY, MESHY_BASE_URL, MESHY_TIMEOUT,
    MESHY_POLL_INTERVAL, MESHY_POLL_TIMEOUT,
)
from agent.models import MeshyResult


def _headers() -> dict:
    return {"Authorization": f"Bearer {MESHY_API_KEY}"}


def create_task(
    prompt: str,
    negative_prompt: str = "low quality, deformed, blurry",
    art_style: str = "realistic",
    topology: str = "quad",
    target_polycount: int = 100_000,
) -> str:
    """Submit a text-to-3D task and return the task_id."""
    payload = {
        "mode": "preview",
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "art_style": art_style,
        "topology": topology,
        "target_polycount": target_polycount,
        "should_remesh": True,
    }
    with httpx.Client(timeout=MESHY_TIMEOUT) as client:
        resp = client.post(
            f"{MESHY_BASE_URL}/v2/text-to-3d",
            json=payload,
            headers=_headers(),
        )
        resp.raise_for_status()
        data = resp.json()
        return data["result"]  # task_id string


def poll_task(task_id: str, progress_callback=None) -> MeshyResult:
    """Poll until task is SUCCEEDED or FAILED. Returns MeshyResult."""
    deadline = time.time() + MESHY_POLL_TIMEOUT
    with httpx.Client(timeout=MESHY_TIMEOUT) as client:
        while time.time() < deadline:
            resp = client.get(
                f"{MESHY_BASE_URL}/v2/text-to-3d/{task_id}",
                headers=_headers(),
            )
            resp.raise_for_status()
            data = resp.json()

            status   = data.get("status", "PENDING")
            progress = data.get("progress", 0)

            if progress_callback:
                progress_callback(progress, status)

            if status == "SUCCEEDED":
                return MeshyResult(
                    task_id=task_id,
                    status="SUCCEEDED",
                    progress=100,
                    model_urls=data.get("model_urls", {}),
                    thumbnail_url=data.get("thumbnail_url"),
                )
            if status == "FAILED":
                return MeshyResult(
                    task_id=task_id,
                    status="FAILED",
                    error_message=data.get("task_error", {}).get("message", "Unknown error"),
                )

            time.sleep(MESHY_POLL_INTERVAL)

    return MeshyResult(
        task_id=task_id,
        status="FAILED",
        error_message="Poll timeout exceeded",
    )


def download_model(result: MeshyResult, dest_dir: str = "meshy_models") -> MeshyResult:
    """Download GLB and OBJ files from a SUCCEEDED MeshyResult. Returns updated result."""
    if result.status != "SUCCEEDED":
        return result

    Path(dest_dir).mkdir(parents=True, exist_ok=True)

    with httpx.Client(timeout=120) as client:
        for fmt in ("glb", "obj"):
            url = result.model_urls.get(fmt)
            if not url:
                continue
            resp = client.get(url)
            resp.raise_for_status()
            dest = Path(dest_dir) / f"{result.task_id}.{fmt}"
            dest.write_bytes(resp.content)
            if fmt == "glb":
                result.local_glb_path = str(dest)
            else:
                result.local_obj_path = str(dest)

    return result
