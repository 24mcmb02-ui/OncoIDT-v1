#!/usr/bin/env bash
# =============================================================================
# OncoIDT — Service Health Check
# Run this after `docker compose up` to verify all services are running.
#
# Usage:
#   bash check_services.sh
#   bash check_services.sh --wait   # wait up to 3 minutes for all services
# =============================================================================

WAIT_MODE=false
if [[ "$1" == "--wait" ]]; then
  WAIT_MODE=true
fi

# Colour codes
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Colour

# Service definitions: "name|url|expected_status"
declare -a SERVICES=(
  "PostgreSQL|pg_isready check|special"
  "Redis|redis_cli check|special"
  "Neo4j Browser|http://localhost:7474|200"
  "MLflow|http://localhost:5000/|200"
  "API Gateway|http://localhost:8000/health|200"
  "Ingestion Service|http://localhost:8001/health|200"
  "Graph Service|http://localhost:8002/health|200"
  "Inference Service|http://localhost:8003/health|200"
  "Reasoner Service|http://localhost:8004/health|200"
  "Alert Service|http://localhost:8005/health|200"
  "Explainability Service|http://localhost:8006/health|200"
  "Simulation Service|http://localhost:8007/health|200"
  "Training Service|http://localhost:8008/health|200"
  "Feature Store Service|http://localhost:8009/health|200"
  "WebSocket Hub|http://localhost:8010/health|200"
  "Frontend|http://localhost:3000|200"
  "Prometheus|http://localhost:9090/-/healthy|200"
  "Grafana|http://localhost:3001/api/health|200"
)

check_http() {
  local name="$1"
  local url="$2"
  local status
  status=$(curl -s -o /dev/null -w "%{http_code}" --max-time 5 "$url" 2>/dev/null)
  if [[ "$status" == "200" || "$status" == "302" ]]; then
    echo -e "  ${GREEN}✓${NC} $name  ${BLUE}($url)${NC}"
    return 0
  else
    echo -e "  ${RED}✗${NC} $name  ${BLUE}($url)${NC}  → HTTP $status"
    return 1
  fi
}

check_postgres() {
  if docker exec oncoidt-postgres pg_isready -U oncoidt -d oncoidt &>/dev/null; then
    echo -e "  ${GREEN}✓${NC} PostgreSQL  (port 5432)"
    return 0
  else
    echo -e "  ${RED}✗${NC} PostgreSQL  (port 5432)  → not ready"
    return 1
  fi
}

check_redis() {
  if docker exec oncoidt-demo-redis redis-cli ping &>/dev/null || \
     docker exec oncoidt-redis redis-cli ping &>/dev/null; then
    echo -e "  ${GREEN}✓${NC} Redis  (port 6379)"
    return 0
  else
    echo -e "  ${RED}✗${NC} Redis  (port 6379)  → not ready"
    return 1
  fi
}

run_checks() {
  local pass=0
  local fail=0

  echo ""
  echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
  echo -e "${BLUE}  OncoIDT Service Health Check${NC}"
  echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
  echo ""

  echo -e "${YELLOW}Infrastructure:${NC}"
  check_postgres && ((pass++)) || ((fail++))
  check_redis && ((pass++)) || ((fail++))
  check_http "Neo4j Browser" "http://localhost:7474" && ((pass++)) || ((fail++))
  check_http "MLflow" "http://localhost:5000/" && ((pass++)) || ((fail++))

  echo ""
  echo -e "${YELLOW}Application Services:${NC}"
  check_http "API Gateway" "http://localhost:8000/health" && ((pass++)) || ((fail++))
  check_http "Ingestion Service" "http://localhost:8001/health" && ((pass++)) || ((fail++))
  check_http "Graph Service" "http://localhost:8002/health" && ((pass++)) || ((fail++))
  check_http "Inference Service" "http://localhost:8003/health" && ((pass++)) || ((fail++))
  check_http "Reasoner Service" "http://localhost:8004/health" && ((pass++)) || ((fail++))
  check_http "Alert Service" "http://localhost:8005/health" && ((pass++)) || ((fail++))
  check_http "Explainability Service" "http://localhost:8006/health" && ((pass++)) || ((fail++))
  check_http "Simulation Service" "http://localhost:8007/health" && ((pass++)) || ((fail++))
  check_http "Training Service" "http://localhost:8008/health" && ((pass++)) || ((fail++))
  check_http "Feature Store Service" "http://localhost:8009/health" && ((pass++)) || ((fail++))
  check_http "WebSocket Hub" "http://localhost:8010/health" && ((pass++)) || ((fail++))

  echo ""
  echo -e "${YELLOW}Frontend & Monitoring:${NC}"
  check_http "Frontend" "http://localhost:3000" && ((pass++)) || ((fail++))
  check_http "Prometheus" "http://localhost:9090/-/healthy" && ((pass++)) || ((fail++))
  check_http "Grafana" "http://localhost:3001/api/health" && ((pass++)) || ((fail++))

  echo ""
  echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
  echo -e "  Result: ${GREEN}$pass passed${NC}  ${RED}$fail failed${NC}  (total: $((pass + fail)))"
  echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
  echo ""

  if [[ $fail -gt 0 ]]; then
    echo -e "${YELLOW}Tip: Check logs for failed services:${NC}"
    echo "  docker compose -f docker-compose.codespaces.yml logs <service-name>"
    echo ""
    echo "  Example:"
    echo "  docker compose -f docker-compose.codespaces.yml logs api-gateway"
    echo ""
  fi

  return $fail
}

if [[ "$WAIT_MODE" == "true" ]]; then
  echo "Waiting for all services to become healthy (up to 3 minutes)..."
  MAX_WAIT=180
  INTERVAL=10
  elapsed=0

  while [[ $elapsed -lt $MAX_WAIT ]]; do
    if run_checks 2>/dev/null | grep -q "0 failed"; then
      echo -e "${GREEN}All services are healthy!${NC}"
      exit 0
    fi
    echo "Retrying in ${INTERVAL}s... (${elapsed}s elapsed)"
    sleep $INTERVAL
    ((elapsed += INTERVAL))
  done

  echo -e "${RED}Timeout: some services did not become healthy within ${MAX_WAIT}s${NC}"
  run_checks
  exit 1
else
  run_checks
  exit $?
fi
