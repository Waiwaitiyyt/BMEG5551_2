# Setting up script for linux VM 

# Install uv
echo "Installing uv..."
curl -LsSf https://astral.sh/uv/install.sh | sh
echo "uv installed successfully."

# Install nodejs
echo "Installing nodejs..."
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.6/install.sh | bash
\. "$HOME/.nvm/nvm.sh"
nvm install 24
node -v
npm -v 
echo "nodejs installed successfully."

# Install claude code
# curl -fsSL https://claude.ai/install.sh | bash
# claude --version