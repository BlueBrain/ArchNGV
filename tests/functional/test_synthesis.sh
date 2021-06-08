set -e

ngv synthesis \
    --config bioname/synthesis.yaml \
    --tns-distributions bioname/tns_distributions.json \
    --tns-parameters bioname/tns_parameters.json \
    --tns-context bioname/tns_context.json \
    --astrocytes build/sonata.tmp/nodes/glia.somata.h5 \
    --microdomains build/microdomains/overlapping_microdomains.h5 \
    --gliovascular-connectivity build/sonata.tmp/edges/gliovascular.connectivity.h5 \
    --neuroglial-connectivity build/sonata.tmp/edges/neuroglial.connectivity.h5 \
    --endfeet-areas build/endfeet_areas.h5 \
    --neuronal-connectivity data/circuit/edges.h5 \
    --out-morph-dir build/morphologies


morph-tool diff build/morphologies/GLIA_0000000000000.h5 expected/morphologies/GLIA_0000000000000.h5
