#!/bin/bash

# 🚀 MicroCurve ML - GitHub Sync Script
# This script helps organize and sync the codebase to GitHub

echo "🧬 MicroCurve ML - GitHub Sync Script"
echo "===================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if git is installed
if ! command -v git &> /dev/null; then
    print_error "Git is not installed. Please install Git first."
    exit 1
fi

# Step 1: Initialize Git repository if not exists
print_status "Step 1: Initializing Git repository..."
if [ ! -d ".git" ]; then
    git init
    print_success "Git repository initialized"
else
    print_success "Git repository already exists"
fi

# Step 2: Clean up temporary files
print_status "Step 2: Cleaning up temporary files..."
rm -f calc *.tmp *.bak *~
print_success "Temporary files cleaned"

# Step 3: Add all files to git
print_status "Step 3: Adding files to Git..."
git add .
print_success "Files added to Git staging area"

# Step 4: Check git status
print_status "Step 4: Checking Git status..."
git status --short

# Step 5: Commit changes
print_status "Step 5: Committing changes..."
read -p "Enter commit message (default: 'Initial commit - MicroCurve ML v1.0'): " commit_msg
if [ -z "$commit_msg" ]; then
    commit_msg="Initial commit - MicroCurve ML v1.0"
fi

git commit -m "$commit_msg"
print_success "Changes committed: $commit_msg"

# Step 6: Set up remote repository
print_status "Step 6: Setting up remote repository..."
echo "Please create a new repository on GitHub first:"
echo "1. Go to https://github.com/new"
echo "2. Repository name: MicroCurve-ML"
echo "3. Description: Advanced Bacterial Growth Prediction using Deep Learning"
echo "4. Make it Public (recommended for open source)"
echo "5. Do NOT initialize with README (we already have one)"
echo ""

read -p "Enter your GitHub repository URL (e.g., https://github.com/username/MicroCurve-ML.git): " repo_url

if [ -n "$repo_url" ]; then
    # Remove existing remote if exists
    git remote remove origin 2>/dev/null || true
    
    # Add new remote
    git remote add origin "$repo_url"
    print_success "Remote repository added: $repo_url"
    
    # Step 7: Push to GitHub
    print_status "Step 7: Pushing to GitHub..."
    git branch -M main
    
    if git push -u origin main; then
        print_success "Successfully pushed to GitHub!"
        echo ""
        echo "🎉 Your MicroCurve ML project is now on GitHub!"
        echo "📍 Repository URL: $repo_url"
        echo ""
        echo "Next steps:"
        echo "1. Add repository description and topics on GitHub"
        echo "2. Enable GitHub Pages for documentation (optional)"
        echo "3. Set up GitHub Actions for CI/CD (optional)"
        echo "4. Add collaborators if needed"
    else
        print_error "Failed to push to GitHub. Please check your credentials and repository URL."
        echo ""
        echo "Troubleshooting:"
        echo "1. Make sure the repository exists on GitHub"
        echo "2. Check your GitHub credentials (git config --global user.name/user.email)"
        echo "3. You might need to authenticate with GitHub (use personal access token)"
    fi
else
    print_warning "No repository URL provided. You can add it later with:"
    echo "git remote add origin <your-repo-url>"
    echo "git push -u origin main"
fi

echo ""
print_status "Repository structure:"
echo "📁 Project files organized and ready for GitHub"
echo "📄 README.md - Comprehensive project documentation"
echo "📄 LICENSE - MIT License"
echo "📄 .gitignore - Git ignore rules"
echo "📄 setup.py - Package installation configuration"
echo "📄 ALGORITHMS.md - Algorithm documentation"
echo "📄 requirements.txt - Python dependencies"

echo ""
print_success "GitHub sync script completed!"
