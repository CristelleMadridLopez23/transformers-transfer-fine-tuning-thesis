"""Entry point for the MatText frozen-transfer workflow, gated by tokenizer integrity."""
from __future__ import annotations

import logging

from provenance.audit_checkpoint import audit


def main() -> None:
    """Run Stage A and halt all dependent stages if tokenizer compatibility fails."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    result = audit()
    if result["status"] != "PASS":
        logging.error("Experiment blocked at critical Stage A; features, fitting, and test remain unrun.")
        raise SystemExit(2)
    logging.info("Stage A passed; later stages are not implemented until the exact tokenizer is verified.")


if __name__ == "__main__":
    main()
