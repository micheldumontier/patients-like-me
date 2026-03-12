#!/bin/bash
# Apply custom QLever UI patches after running `qlever ui`.
# Usage: cd qlever && qlever ui && bash setup-ui.sh

CONTAINER="qlever.ui.mimic-iv-demo"

echo "Injecting IRI browse script into QLever UI..."

# Copy custom JS into the container
docker cp "$(dirname "$0")/iri-browse.js" "$CONTAINER:/app/backend/static/js/iri-browse.js"

# Add script tag if not already present
docker exec "$CONTAINER" bash -c '
  if ! grep -q "iri-browse.js" /app/backend/templates/partials/head.html; then
    sed -i "s|<script src=\"{% static \"js/qleverUI.js\" %}\"></script>|<script src=\"{% static \\\"js/qleverUI.js\\\" %}\"></script>\n  <script src=\"{% static \\\"js/iri-browse.js\\\" %}\"></script>|" /app/backend/templates/partials/head.html
    echo "  Script tag added."
  else
    echo "  Script tag already present."
  fi
'

echo "Done. Hard-refresh the browser (Cmd+Shift+R) to pick up the changes."
