# frankenstein

**Task:** you have to predict text from brain activity.

**Data:** there were recorded 12100 sentences on one patient, who can no longer speak intelligibly due amyotrophic lateral sclerosis. (input):  Utah array with 256 electrodes. (output): text sentences.

**Deadline:** June 2, 2024

**Details:** perhaps competition at ICML 24 

**Link:** [https://eval.ai/web/challenges/challenge-page/2099/overview](https://eval.ai/web/challenges/challenge-page/2099/overview)



### Architecture

**North Star:**

VQVAE -> MAE -> Perciever (projector) ->  LLAMA 3. 