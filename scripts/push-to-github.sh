#!/bin/bash
set -e

REPO_NAME="agentkernel"
GITHUB_USER="HuuBar"

echo "=== AgentKernel GitHub Push Script ==="
echo ""

if [ -z "$GITHUB_TOKEN" ]; then
    echo "Please set your GitHub Personal Access Token:"
    echo "  export GITHUB_TOKEN=ghp_xxxxxxxx"
    echo ""
    echo "Or enter it now:"
    read -s GITHUB_TOKEN
    export GITHUB_TOKEN
fi

echo "Creating GitHub repository if not exists..."
curl -s -H "Authorization: token $GITHUB_TOKEN"      -H "Accept: application/vnd.github.v3+json"      https://api.github.com/repos/$GITHUB_USER/$REPO_NAME      || true

REMOTE_URL="https://$GITHUB_TOKEN@github.com/$GITHUB_USER/$REPO_NAME.git"
git remote add origin $REMOTE_URL 2>/dev/null || git remote set-url origin $REMOTE_URL

git push -u origin main --force

echo ""
echo "=== Done! ==="
echo "Repository: https://github.com/$GITHUB_USER/$REPO_NAME"
