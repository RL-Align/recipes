ARG BASE_IMAGE
FROM ${BASE_IMAGE}
USER root
ARG RLK_SHA
ARG VERL_SHA
ARG RLK_TREE
ARG VERL_TREE
ARG RECIPE_COMMIT
LABEL org.opencontainers.image.source="https://github.com/RL-Align/recipes" \
      org.opencontainers.image.revision="${RECIPE_COMMIT}" \
      rl-align.validation.scope="seven-layer-actor-tp1-experimental-gpu-unvalidated"
RUN set -eu; \
    rm -rf /opt/dsv4/RL-Kernel /opt/dsv4/verl; \
    for repo in RL-Kernel verl; do \
      if [ "$repo" = RL-Kernel ]; then source_sha="$RLK_SHA"; source_tree="$RLK_TREE"; \
      else source_sha="$VERL_SHA"; source_tree="$VERL_TREE"; fi; \
      git init "/opt/dsv4/$repo"; \
      git -C "/opt/dsv4/$repo" remote add origin "https://github.com/RL-Align/$repo.git"; \
      git -C "/opt/dsv4/$repo" fetch --depth=1 origin "$source_sha"; \
      git -C "/opt/dsv4/$repo" checkout --detach FETCH_HEAD; \
      test "$(git -C "/opt/dsv4/$repo" rev-parse HEAD)" = "$source_sha"; \
      test "$(git -C "/opt/dsv4/$repo" rev-parse HEAD^{tree})" = "$source_tree"; \
      test -z "$(git -C "/opt/dsv4/$repo" status --porcelain)"; \
    done
COPY scripts/seven_layer /opt/recipes/scripts/seven_layer
COPY scripts/node.py /opt/recipes/scripts/node.py
COPY scripts/binary_manifest.py /opt/recipes/scripts/binary_manifest.py
COPY models/deepseek-v4/rocm/seven-layer /opt/recipes/models/deepseek-v4/rocm/seven-layer
ENV PATH="/opt/dsv4-venv/bin:${PATH}"
ENV PYTHONPATH="/opt/dsv4/RL-Kernel:/opt/dsv4/verl"
RUN test -r /opt/dsv4/lib/rlk-isolated-kfd.so && \
    mkdir -p /opt/recipe-build && \
    python /opt/recipes/scripts/binary_manifest.py > /opt/recipe-build/binaries.json
WORKDIR /opt/recipes
