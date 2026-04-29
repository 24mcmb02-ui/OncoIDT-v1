#!/usr/bin/env bash
# =============================================================================
# OncoIDT — Development mTLS Certificate Generator
#
# Generates a local CA and per-service TLS certificates for development.
# All certs are output to infra/certs/dev/
#
# Usage:
#   chmod +x infra/certs/generate_dev_certs.sh
#   ./infra/certs/generate_dev_certs.sh
#
# Requirements: openssl
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${SCRIPT_DIR}/dev"
CA_KEY="${OUTPUT_DIR}/ca.key"
CA_CERT="${OUTPUT_DIR}/ca.crt"
DAYS_VALID=365

SERVICES=(
  "api-gateway"
  "ingestion-service"
  "graph-service"
  "inference-service"
  "reasoner-service"
  "alert-service"
  "explainability-service"
  "simulation-service"
  "training-service"
  "feature-store-service"
  "websocket-hub"
)

# ---------------------------------------------------------------------------
# Idempotent: skip if certs already exist and are still valid
# ---------------------------------------------------------------------------
if [[ -f "${CA_CERT}" ]]; then
  if openssl x509 -checkend 86400 -noout -in "${CA_CERT}" 2>/dev/null; then
    echo "CA certificate already exists and is valid. Skipping CA generation."
    SKIP_CA=true
  else
    echo "CA certificate exists but is expired or expiring soon. Regenerating."
    SKIP_CA=false
  fi
else
  SKIP_CA=false
fi

mkdir -p "${OUTPUT_DIR}"

# ---------------------------------------------------------------------------
# 1. Generate CA key and self-signed certificate
# ---------------------------------------------------------------------------
if [[ "${SKIP_CA}" == "false" ]]; then
  echo "Generating CA key and certificate..."
  openssl genrsa -out "${CA_KEY}" 4096

  openssl req -new -x509 \
    -key "${CA_KEY}" \
    -out "${CA_CERT}" \
    -days "${DAYS_VALID}" \
    -subj "/C=US/ST=Dev/L=Dev/O=OncoIDT/OU=CA/CN=oncoidt-dev-ca"

  echo "CA certificate generated: ${CA_CERT}"
fi

# ---------------------------------------------------------------------------
# 2. Generate per-service key, CSR, and signed certificate
# ---------------------------------------------------------------------------
for SERVICE in "${SERVICES[@]}"; do
  SERVICE_DIR="${OUTPUT_DIR}/${SERVICE}"
  SERVICE_KEY="${SERVICE_DIR}/service.key"
  SERVICE_CSR="${SERVICE_DIR}/service.csr"
  SERVICE_CERT="${SERVICE_DIR}/service.crt"
  EXT_FILE="${SERVICE_DIR}/ext.cnf"

  # Skip if cert already exists and is valid
  if [[ -f "${SERVICE_CERT}" ]]; then
    if openssl x509 -checkend 86400 -noout -in "${SERVICE_CERT}" 2>/dev/null; then
      echo "Certificate for ${SERVICE} already exists and is valid. Skipping."
      continue
    fi
  fi

  mkdir -p "${SERVICE_DIR}"

  echo "Generating certificate for ${SERVICE}..."

  # Generate service private key
  openssl genrsa -out "${SERVICE_KEY}" 2048

  # Create extension config with SAN
  cat > "${EXT_FILE}" <<EOF
[req]
req_extensions = v3_req
distinguished_name = req_distinguished_name

[req_distinguished_name]

[v3_req]
subjectAltName = @alt_names
keyUsage = digitalSignature, keyEncipherment
extendedKeyUsage = serverAuth, clientAuth

[alt_names]
DNS.1 = ${SERVICE}
DNS.2 = ${SERVICE}.oncoidt
DNS.3 = ${SERVICE}.oncoidt.svc.cluster.local
DNS.4 = localhost
IP.1 = 127.0.0.1
EOF

  # Generate CSR
  openssl req -new \
    -key "${SERVICE_KEY}" \
    -out "${SERVICE_CSR}" \
    -subj "/C=US/ST=Dev/L=Dev/O=OncoIDT/OU=Services/CN=${SERVICE}" \
    -config "${EXT_FILE}"

  # Sign with CA
  openssl x509 -req \
    -in "${SERVICE_CSR}" \
    -CA "${CA_CERT}" \
    -CAkey "${CA_KEY}" \
    -CAcreateserial \
    -out "${SERVICE_CERT}" \
    -days "${DAYS_VALID}" \
    -extensions v3_req \
    -extfile "${EXT_FILE}"

  # Clean up CSR and ext file
  rm -f "${SERVICE_CSR}" "${EXT_FILE}"

  echo "  Key:  ${SERVICE_KEY}"
  echo "  Cert: ${SERVICE_CERT}"
done

# ---------------------------------------------------------------------------
# 3. Print usage instructions
# ---------------------------------------------------------------------------
echo ""
echo "============================================================"
echo "OncoIDT Dev Certificates Generated"
echo "============================================================"
echo ""
echo "CA Certificate:  ${CA_CERT}"
echo "CA Key:          ${CA_KEY}"
echo ""
echo "Per-service certificates are in: ${OUTPUT_DIR}/<service-name>/"
echo "  service.key  — private key"
echo "  service.crt  — certificate signed by the dev CA"
echo ""
echo "Usage in Docker Compose:"
echo "  Mount the certs directory as a volume:"
echo "    volumes:"
echo "      - ./infra/certs/dev/<service-name>:/certs:ro"
echo ""
echo "  Set environment variables:"
echo "    MTLS_CA_CERT_PATH=/certs/ca.crt"
echo "    MTLS_SERVICE_CERT_PATH=/certs/service.crt"
echo "    MTLS_SERVICE_KEY_PATH=/certs/service.key"
echo ""
echo "Usage in Kubernetes:"
echo "  Apply cert-manager config:"
echo "    kubectl apply -f infra/certs/cert-manager-config.yaml"
echo ""
echo "WARNING: These are development certificates only."
echo "         Do NOT use in production."
echo "============================================================"
