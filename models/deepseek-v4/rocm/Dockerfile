ARG BASE_IMAGE
FROM ${BASE_IMAGE}
USER root
ARG RLK_SHA
ARG VERL_SHA
ARG RLK_TREE
ARG VERL_TREE
ARG GPU_ARCH=gfx942
ARG RECIPE_COMMIT
LABEL org.opencontainers.image.source="https://github.com/RL-Align/recipes" \
      org.opencontainers.image.revision="${RECIPE_COMMIT}"
COPY models/deepseek-v4/rocm/patches /opt/recipe-patches
RUN set -eu; \
    mkdir -p /opt/recipe; \
    git init /opt/recipe/RL-Kernel; \
    git -C /opt/recipe/RL-Kernel remote add origin https://github.com/RL-Align/RL-Kernel.git; \
    git -C /opt/recipe/RL-Kernel fetch --depth=1 origin "$RLK_SHA"; \
    git -C /opt/recipe/RL-Kernel checkout --detach FETCH_HEAD; \
    git -C /opt/recipe/RL-Kernel apply /opt/recipe-patches/eight-gpu-rl-kernel.patch; \
    git -C /opt/recipe/RL-Kernel apply /opt/recipe-patches/recipe-rl-kernel.patch; \
    git -C /opt/recipe/RL-Kernel add -A; \
    test "$(git -C /opt/recipe/RL-Kernel write-tree)" = "$RLK_TREE"; \
    git init /opt/recipe/verl; \
    git -C /opt/recipe/verl remote add origin https://github.com/RL-Align/verl.git; \
    git -C /opt/recipe/verl fetch --depth=1 origin "$VERL_SHA"; \
    git -C /opt/recipe/verl checkout --detach FETCH_HEAD; \
    git -C /opt/recipe/verl apply /opt/recipe-patches/eight-gpu-verl.patch; \
    git -C /opt/recipe/verl apply /opt/recipe-patches/recipe-verl.patch; \
    git -C /opt/recipe/verl add -A; \
    test "$(git -C /opt/recipe/verl write-tree)" = "$VERL_TREE"; \
    for repo in RL-Kernel verl; do \
      GIT_AUTHOR_DATE=2026-11-25T00:00:00Z GIT_COMMITTER_DATE=2026-11-25T00:00:00Z \
      git -C "/opt/recipe/$repo" -c user.name='RL-Align Recipes' \
        -c user.email='recipes@users.noreply.github.com' commit -m 'Apply locked DSV4 recipe patches'; \
    done; \
    test -r /opt/dsv4/lib/rlk-isolated-kfd.so; \
    rm -rf /opt/dsv4/RL-Kernel /opt/dsv4/verl; \
    mv /opt/recipe/RL-Kernel /opt/dsv4/RL-Kernel; \
    mv /opt/recipe/verl /opt/dsv4/verl
COPY . /opt/recipes
ENV PATH="/opt/dsv4-venv/bin:${PATH}"
ENV PYTHONPATH="/opt/dsv4/RL-Kernel:/opt/dsv4/verl:${PYTHONPATH}"
RUN if [ "$GPU_ARCH" = gfx950 ]; then /bin/bash /opt/recipes/scripts/rebuild_gfx950.sh; \
    else test "$GPU_ARCH" = gfx942; fi
RUN mkdir -p /opt/recipe-build && \
    /opt/dsv4-venv/bin/python /opt/recipes/scripts/binary_manifest.py > /opt/recipe-build/binaries.json
WORKDIR /opt/recipes
