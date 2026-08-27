"""Shape, type and sampling of one parameter, in the plain-text form AMDA_Kernel reads.

AMDA_Kernel needs four numbers before it can allocate anything for a parameter, and it
asks for them one parameter at a time, from C++, over curl. /get_inventory carries the
same information but serialises a whole provider per call, which is not something that
client can afford once per parameter.

The route name and the ``KEY=value`` body are AMDA_Kernel's existing contract
(``GetVIWithSpeasyNode.cc``), not a design choice: the kernel scans the response for the
four keys and falls back on its own defaults for any key it does not find.
"""
import logging
from typing import Optional, Tuple

import speasy as spz
from fastapi import Query
from fastapi.responses import PlainTextResponse
from starlette.concurrency import run_in_threadpool

from .routes import router

log = logging.getLogger(__name__)

_INT_CDF_TYPES = ("INT", "BYTE")


def _amda_type(cdf_type) -> str:
    return "INT" if any(t in str(cdf_type or "").upper() for t in _INT_CDF_TYPES) else "DOUBLE"


def _dims(spz_shape) -> Optional[Tuple[int, int]]:
    """(DIM1, DIM2) from speasy's inventory shape: 1 for a scalar, else the shape past the record axis."""
    if spz_shape is None:
        return None
    if isinstance(spz_shape, int):
        return 1, 1
    dims = list(spz_shape) + [1, 1]
    return int(dims[0]), int(dims[1])


def _inventory_nodes(path: str):
    """The (parameter, dataset) index nodes behind a speasy UID; either may be None."""
    provider, _, uid = path.partition("/")
    inventory = getattr(spz.inventories.flat_inventories, provider, None)
    if inventory is None or "/" not in uid:
        return None, None
    return inventory.parameters.get(uid), inventory.datasets.get(uid.rsplit("/", 1)[0])


def _metadata_lines(path: str) -> str:
    param, dataset = _inventory_nodes(path)
    if param is None:
        return ""
    lines = [f"TYPE={_amda_type(getattr(param, 'cdf_type', None))}"]
    dims = _dims(getattr(param, "spz_shape", None))
    if dims is not None:
        lines += [f"DIM1={dims[0]}", f"DIM2={dims[1]}"]
    min_sampling = getattr(dataset, "MinSampling", None) if dataset is not None else None
    if min_sampling is not None:
        # AMDA_Kernel wants seconds. It multiplies this by its gap threshold to widen the
        # interval it asks for, so a value in the wrong unit makes it request centuries.
        lines.append(f"MIN_SAMPLING={float(min_sampling)}")
    return "\n".join(lines) + "\n"


@router.get('/metadata', response_class=PlainTextResponse,
            description='Parameter type, dimensions and sampling, as AMDA_Kernel expects them')
async def metadata(path: str = Query(examples=["archive/ddbase/ace_imf_all/IMF"])):
    log.debug(f'Metadata request for {path}')
    body = await run_in_threadpool(_metadata_lines, path)
    if not body:
        log.debug(f'{path} is not in any inventory')
        return PlainTextResponse(content=f"Unknown product: {path}\n", status_code=404)
    return PlainTextResponse(content=body)
