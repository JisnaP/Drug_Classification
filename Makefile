install:
	pip install --upgrade pip && \
	pip install -r requirements.txt

format:
	black *.py

train: 
	python train.py

eval:   
	echo "## Model Metrics" >> report.md && \
	cat ./results/metrics.txt >> report.md && \
	echo '\n ## Confusion Matrix Plot ' >> report.md && \
	echo '![Confusion Matrix](./results/model_results.png)' >> report.md
	
update-branch:
	git config --global user.name $(USER_NAME) && \
	git config --global user.email $(USER_EMAIL) && \
	git add . && \
	git commit -m "Update with new results" && \
	git push --force origin HEAD:update

hf-login:
	pip install -U "huggingface_hub[cli]" && \
	git pull origin update && \
	git switch update && \
	huggingface-cli login --token $(HF) --add-to-git-credential

push-hub:
	huggingface-cli upload Jisna12/Drug Classification ./App --repo-type=space --commit-message="Sync App files" && \
	huggingface-cli upload Jisna12/Drug Classification ./model /model --repo-type=space --commit-message="Sync model" && \
	huggingface-cli upload Jisna12/Drug Classification ./results /metrics --repo-type=space --commit-message="Sync results"

deploy: hf-login push-hub

all: install format train eval update-branch deploy
