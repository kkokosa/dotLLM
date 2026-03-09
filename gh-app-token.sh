  #!/usr/bin/env bash
  # Usage: source ~/.dotllm/gh-app-token.sh
  # Sets CLAUDE_GH_TOKEN to a fresh GitHub App installation token.

  APP_ID="2989778"
  PEM="$HOME/.dotllm/dotllm-claude-code-bot.2026-03-02.private-key.pem"

  now=$(date +%s)
  iat=$((now - 60))
  exp=$((now + 540))

  b64url() { openssl base64 -e | tr -d '=' | tr '/+' '_-' | tr -d '\n\r'; }

  header=$(printf '{"alg":"RS256","typ":"JWT"}' | b64url)
  payload=$(printf '{"iat":%d,"exp":%d,"iss":"%s"}' "$iat" "$exp" "$APP_ID" | b64url)
  unsigned="${header}.${payload}"
  sig=$(printf '%s' "$unsigned" | openssl dgst -sha256 -sign "$PEM" | b64url)
  JWT="${unsigned}.${sig}"

  INST_ID=$(curl -s --ssl-no-revoke \
    -H "Authorization: Bearer $JWT" \
    -H "Accept: application/vnd.github+json" \
    https://api.github.com/app/installations | python -c "import sys,json;print(json.load(sys.stdin)[0]['id'])")

  export CLAUDE_GH_TOKEN=$(curl -s --ssl-no-revoke -X POST \
    -H "Authorization: Bearer $JWT" \
    -H "Accept: application/vnd.github+json" \
    "https://api.github.com/app/installations/$INST_ID/access_tokens" | python -c "import sys,json;print(json.load(sys.stdin)['token'])")

  echo "CLAUDE_GH_TOKEN set (expires in 1h), installation: $INST_ID"