.PHONY: verify test audit replay
verify:
	python scripts/verify_package.py
	python scripts/verify_v14.py
test:
	python -m pytest -q
audit:
	python scripts/review_public_tree.py
replay:
	python scripts/rebuild_v14.py --output $(REPLAY_DIR)
