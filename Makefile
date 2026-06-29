MODEL_0_6B = /Users/mizzy/models/Qwen3-0.6B
MODEL_1_7B = /Users/mizzy/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e
PROMPT ?= <|im_start|>user\nWhat is 1+1?<|im_end|>\n<|im_start|>assistant\n

.PHONY: run-0.6b run-1.7b bench-0.6b bench-1.7b bench-0.6b-q8 bench-1.7b-q8

run-0.6b:
	cargo run --release -- $(MODEL_0_6B) $$'$(PROMPT)'

run-1.7b:
	cargo run --release -- $(MODEL_1_7B) $$'$(PROMPT)'

bench-0.6b:
	cargo run --release -- $(MODEL_0_6B) $$'$(PROMPT)' --bench

bench-1.7b:
	cargo run --release -- $(MODEL_1_7B) $$'$(PROMPT)' --bench

bench-0.6b-q8:
	cargo run --release -- $(MODEL_0_6B) $$'$(PROMPT)' --bench --quantize

bench-1.7b-q8:
	cargo run --release -- $(MODEL_1_7B) $$'$(PROMPT)' --bench --quantize
