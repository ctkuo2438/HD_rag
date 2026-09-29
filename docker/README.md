# Human Design RAG - Docker guide

- [For users](#for-users): download and run the app with your own API keys.
- [Maintainer build and release](#maintainer-build-and-release): build images,
  publish to GHCR, and create the starter archive.

## For users

Install Docker Desktop (or Docker Engine with Compose). Extract the starter archive
into a folder. You only need its three files; no Git clone, Python installation,
PDF upload, or index build is required.

1. Copy `.env.example` to `.env` and fill in your own `OPENAI_API_KEY`.
2. Run these commands in the extracted folder:

   ```sh
   docker compose pull
   docker compose up -d
   ```

3. Open http://127.0.0.1:8501. Ask one focused question and optionally upload a
   BodyGraph image. Stop another app using port 8501 first.

The image includes the complete Chroma/BM25 knowledge index and model defaults.
The first download can take a few minutes. Your API account must have access to
the configured models. The image does not include model weights; questions still
use provider APIs.

### Costs and keys

This launcher opts into paid OpenAI calls when you submit a question. Opening the
page makes no paid calls. Your keys stay in your local setup and are used to call
the providers directly; there is no project-hosted server. Keep `.env` private.
OpenAI receives your question, selected passages/chart facts, and any uploaded
image. The first image extraction can take several minutes; subsequent questions
reuse the validated chart within the same browser session.

Cohere reranking is optional and disabled by default. To enable it, set these in
`.env` (its dependency and model default are already provided):

```dotenv
COHERE_API_KEY=replace-with-your-cohere-key
HD_RAG_RERANK_PROVIDER=cohere
HD_RAG_REAL_RERANK_API=1
```

Cohere receives the question and candidate passages and charges separately.
Process-environment values override `.env`. After changing settings, run
`docker compose up -d` again; `restart` alone does not apply new settings.

### Stop

```sh
docker compose down
```

The knowledge index remains in the downloaded image. No local `storage/` folder
or volume is required. Answers are reflective information, not medical, legal,
or financial advice. Full-chart readings are not supported.

## Maintainer build and release

Run the following commands from the **source repository root**. They require the
source code and an existing local hybrid index. Users of the prebuilt image only
need the instructions above.

There are two release artifacts:

- **Docker image**: `ghcr.io/ctkuo2438/hd-rag:0.1.0`, including the app and the
  complete knowledge index. Publish this to GHCR.
- **Starter archive**: `dist/hd-rag-starter-0.1.0.tar.gz`, containing Compose,
  the key template and this guide. Attach this to a GitHub Release.

The image distributes indexed book passages as well as vectors. Keep keys,
original PDFs, private chart images and unrelated storage out of the image and Git.

### 1. Build and verify the image

Stop processes using the source index before copying it into a release. Keep all
four artifacts together: `manifest.json`, `nodes.jsonl`, `chroma/`, and `bm25/`.

```sh
docker buildx build --target release \
  --build-context knowledge=./storage/hybrid_v1 \
  --build-arg VERSION=0.1.0 \
  --platform linux/amd64,linux/arm64 \
  --tag ghcr.io/ctkuo2438/hd-rag:0.1.0 --load .
```

Multi-platform local loading requires Docker's containerd image store. This build
creates both Apple Silicon and Intel/AMD variants with the same knowledge corpus.
It copies the existing index and never rebuilds embeddings. The release stage
verifies canonical IDs and Chroma/BM25 identity/counts without network access.
Smoke-test both architectures with all real provider flags disabled before publishing.

The ordinary source build context excludes `.env`, PDFs and storage. The explicit
`knowledge` context supplies only the existing index artifacts copied by the Dockerfile.

### 2. Push the local image to GHCR

Use a GitHub personal access token (classic) with `write:packages` permission.
Enter it at Docker's password prompt; do not put it in a command argument or image.
Once the local image has passed verification, run:

```sh
docker login ghcr.io --username ctkuo2438
docker push ghcr.io/ctkuo2438/hd-rag:0.1.0
```

This uploads the already-built local image. Keep both architectures by using the
default push; `--platform` would publish a single architecture instead. See
[Docker's push documentation](https://docs.docker.com/reference/cli/docker/image/push/).

Set the package visibility to **Public** in GitHub's package settings, then check
that a client without registry credentials can pull it. Newly created GHCR packages
are private by default. See
[GitHub's registry documentation](https://docs.github.com/en/packages/working-with-a-github-packages-registry/working-with-the-container-registry).

### 3. Create the starter archive

```sh
mkdir -p dist
COPYFILE_DISABLE=1 tar -czf dist/hd-rag-starter-0.1.0.tar.gz compose.yaml -C docker .env.example README.md
```

The archive contains only three files: a copy of the root `compose.yaml`, plus
`docker/.env.example` and `docker/README.md` renamed to `.env.example` and `README.md`.
It contains no image or index. `COPYFILE_DISABLE=1` prevents macOS metadata files
from being added to the archive. The generated `dist/` directory is ignored by Git.

Attach the archive to the matching GitHub Release and add its download link to the
project README. Regenerate it whenever one of its three source files changes.
Keep published image tags immutable; use a new version when code, models or the
knowledge corpus change, and update the Compose tag and archive together.

### Optional local development

To test source changes with a host-mounted index, configure the repository's `.env`
from its root `.env.example`, then run:

```sh
docker compose -f compose.local.yaml up --build -d
```

This builds the Dockerfile's `app` stage and mounts the local index selected by
`HD_RAG_INDEX_DIR` (default `storage/hybrid_v1`). It does not bundle the index into
the image. `compose.local.yaml` is for development only; it is not needed by users
of the release image and is not included in the starter archive.
