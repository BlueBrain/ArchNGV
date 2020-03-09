snakemake --snakefile ./Snakefile \
          --config bioname=../bioname \
          --directory ./build \
          --cluster-config ./bioname/cluster.yaml \
          -F all

