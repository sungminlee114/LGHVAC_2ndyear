apt update
apt install gh
gh auth login

git clone https://github.com/sungminlee114/LGHVAC_2ndyear
cd LGHVAC_2ndyear
git config --global user.email "$(gh api user --jq '.email')" && \
git config --global user.name "$(gh api user --jq '.login')"
git checkout main

pip install unsloth peft transformers jupyter ipywidgets

# kill -9 $(ps aux | grep 'jupyter' | grep -v grep | awk '{print $2}')