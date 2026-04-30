#!/bin/bash
# AgentKernel — 一键推送到 GitHub
# Usage: ./scripts/push-to-github.sh [GITHUB_TOKEN]

set -e

REPO_NAME="agentkernel"
GITHUB_USER="HuuBar"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

echo "============================================================"
echo "  AgentKernel — GitHub Push Script"
echo "  Target: https://github.com/$GITHUB_USER/$REPO_NAME"
echo "============================================================"
echo ""

# 检查是否在正确的目录
cd "$REPO_ROOT"
if [ ! -f "pyproject.toml" ]; then
    echo "❌ Error: Please run this script from the repository root"
    exit 1
fi

# 获取 GitHub Token
if [ -n "$1" ]; then
    TOKEN="$1"
elif [ -n "$GITHUB_TOKEN" ]; then
    TOKEN="$GITHUB_TOKEN"
else
    echo "🔑 GitHub Personal Access Token (classic) required."
    echo "   Create one at: https://github.com/settings/tokens/new"
    echo "   Required scopes: repo (full control of private repositories)"
    echo ""
    echo "   Enter your token (input will be hidden):"
    read -rs TOKEN
    echo ""
fi

if [ -z "$TOKEN" ]; then
    echo "❌ No token provided. Exiting."
    exit 1
fi

# 验证 token
echo "🔍 Verifying GitHub token..."
USER_RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" \
    -H "Authorization: Bearer $TOKEN" \
    -H "Accept: application/vnd.github.v3+json" \
    https://api.github.com/user)

if [ "$USER_RESPONSE" != "200" ]; then
    echo "❌ Invalid token or network error (HTTP $USER_RESPONSE)"
    exit 1
fi

USERNAME=$(curl -s -H "Authorization: Bearer $TOKEN" \
    https://api.github.com/user | grep -o '"login":"[^"]*"' | cut -d'"' -f4)
echo "   ✅ Authenticated as: $USERNAME"

# 创建仓库（如果不存在）
echo ""
echo "📦 Creating GitHub repository if not exists..."
REPO_RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" \
    -H "Authorization: Bearer $TOKEN" \
    -H "Accept: application/vnd.github.v3+json" \
    https://api.github.com/repos/$GITHUB_USER/$REPO_NAME)

if [ "$REPO_RESPONSE" = "404" ]; then
    echo "   Creating new repository..."
    CREATE_RESPONSE=$(curl -s -w "\n%{http_code}" \
        -X POST \
        -H "Authorization: Bearer $TOKEN" \
        -H "Accept: application/vnd.github.v3+json" \
        -d "{\"name\":\"$REPO_NAME\",\"description\":\"Budget-Aware Call Graph (BACG) — Topology-aware resource allocation for AI Agents\",\"private\":false,\"has_issues\":true,\"has_wiki\":false}" \
        https://api.github.com/user/repos)
    HTTP_CODE=$(echo "$CREATE_RESPONSE" | tail -n1)
    if [ "$HTTP_CODE" = "201" ]; then
        echo "   ✅ Repository created: https://github.com/$GITHUB_USER/$REPO_NAME"
    else
        echo "   ⚠️  Repository creation response: HTTP $HTTP_CODE"
        echo "   You may need to create it manually at:"
        echo "   https://github.com/new?repo_name=$REPO_NAME"
    fi
else
    echo "   ✅ Repository already exists"
fi

# 设置 remote
echo ""
echo "🔗 Configuring git remote..."
REMOTE_URL="https://$TOKEN@github.com/$GITHUB_USER/$REPO_NAME.git"
git remote remove origin 2>/dev/null || true
git remote add origin "$REMOTE_URL"
echo "   ✅ Remote configured: github.com/$GITHUB_USER/$REPO_NAME"

# 推送
echo ""
echo "🚀 Pushing to GitHub..."
if git push -u origin main --force; then
    echo ""
    echo "============================================================"
    echo "  ✅ Successfully pushed to GitHub!"
    echo ""
    echo "  Repository: https://github.com/$GITHUB_USER/$REPO_NAME"
    echo "  Website:    https://$GITHUB_USER.github.io/$REPO_NAME/"
    echo ""
    echo "  Next steps:"
    echo "    1. Visit the repository to enable GitHub Pages"
    echo "       Settings → Pages → Source: GitHub Actions"
    echo "    2. Add OPENAI_API_KEY secret for benchmarks"
    echo "       Settings → Secrets and variables → Actions"
    echo "    3. Run benchmarks locally: make benchmark"
    echo "============================================================"
else
    echo ""
    echo "❌ Push failed. Common fixes:"
    echo "   1. Check your token has 'repo' scope"
    echo "   2. Ensure the repository name doesn't conflict"
    echo "   3. Try manual push:"
    echo "      git remote add origin https://github.com/$GITHUB_USER/$REPO_NAME.git"
    echo "      git push -u origin main"
    echo ""
    echo "   Or use the GitHub CLI:"
    echo "      gh auth login"
    echo "      gh repo create $GITHUB_USER/$REPO_NAME --public --push --source=."
    exit 1
fi
