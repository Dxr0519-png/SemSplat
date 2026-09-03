# semsplat helper targets.
# All training must export the proxy so the HF CLIP weights and viewer work:
#   make PROXY=http://127.0.0.1:7897 train-data ...
PROXY ?= http://127.0.0.1:7897
PY = .venv/bin/python
NS = .venv/bin/ns-train
export https_proxy := $(PROXY)
export http_proxy := $(PROXY)

.PHONY: env torch gsplat nerfstudio install smoke mini mini-train test \
        data-demo demo-train query train help clean

help:
	@echo "targets: env gsplat nerfstudio install mini mini-train smoke test data-demo demo-train query"

## environment bring-up (order matters)
env:          ## venv + torch cu128 (slow)
	scripts/build_env.sh
gsplat:       ## source-build gsplat==1.4.0 for sm_120 (needs CUDA_HOME->12.8+)
	scripts/build_gsplat.sh
nerfstudio:   ## install nerfstudio 1.1.5 + this package
	scripts/install_nerfstudio.sh
install: env gsplat nerfstudio

## smoke test
smoke:        ## registration + 30-iter train on mini scene
	scripts/smoke_plugin.sh

mini:         ## synth a tiny nerfstudio scene for smoke tests
	$(PY) scripts/make_mini_scene.py --out /tmp/mini_scene

mini-train:
	$(NS) semsplat --data /tmp/mini_scene \
	    --max-num-iterations 60 --viewer.quit-on-train-completion True \
	    --viewer.websocket-port 7007

## data (Replica)
data-demo:    ## download + convert NICE-SLAM Demo scene
	scripts/prepare_replica.sh --url https://cvg-data.inf.ethz.ch/nice-slam/data/Demo.zip --out data/replica_demo

demo-train:
	$(NS) semsplat-replica --data data/replica_demo \
	    --max-num-iterations 6000 --viewer.quit-on-train-completion True \
	    --viewer.websocket-port 7007 --output-dir outputs/demo

test:
	$(PY) -m pytest tests/ -x -q

clean:
	rm -rf .venv outputs results data
