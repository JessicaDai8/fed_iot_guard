# Byzantine Robustness for Fed_IoT_Guard (https://github.com/ValerianRey/fed_iot_guard)

## About This Repo

In 2022, researchers from École Polytechnique Fédérale de Lausanne developed an federated learning (FL) framework for detecting malware in IoT devices (Rey, 2022). Their paper, however, notes that their algorithm is vulnerable to data poisoning attacks: attacks characterized by sending mislabeled data to intentionally disrupt the model (Rey, 2022; Tolpegin, 2020).

Their paper proposed to resolve this issue through the Byzantine-Robust Aggregation algorithms -- algorithms that no matter how many malicious, or "Byzantine" nodes send improper data to the central server, the gradient for the central model will still converge (in gradient descent) (Blanchard et al, 2017).

In 2023, researchers from Berkeley (among other universities) released a paper formulating Byzantine Robust protocols for FL (Zhu, 2023). Their code is accessible in this repo (https://github.com/wanglun1996/secure-robust-federated-learning).

Our project seeks to improve Fed_IoT_Guard's resistance to data poisoning attacks via the Byzantine-Robust Aggregation methods from (Zhu, 2023). Doing so can help organizations worldwide more effectively avoid malware threats, reducing financial loss.

## Testing Code
To test the three different aggregation functions, go to main.py and change line 81 to your desired aggregated function. You can choose betweek krum_aggregation, federated_averaging and federated_median. Remember that Krum's is available onnly in the Krum's branch.

To simulate our results, run these commands:

Supervised: 
python src/main.py decentralized classifier --test --fedavg --collaborative --verbose-depth=5

Unsupervised:
python src/main.py decentralized autoencoder --test --fedavg --collaborative --verbose-depth=6


# Works Cited

Blanchard, P., El Mhamdi, E. M., Guerraoui, R., & Stainer, J. (2017). Machine learning with adversaries: Byzantine tolerant gradient descent. Advances in neural information processing systems, 30.

Rey, V., Sánchez, P. M. S., Celdrán, A. H., & Bovet, G. (2022). Federated learning for malware detection in IoT devices. Computer Networks, 204, 108693.

Tolpegin, V., Truex, S., Gursoy, M. E., & Liu, L. (2020). Data poisoning attacks against federated learning systems. In Computer Security–ESORICS 2020: 25th European Symposium on Research in Computer Security, ESORICS 2020, Guildford, UK, September 14–18, 2020, Proceedings, Part I 25 (pp. 480-501). Springer International Publishing.

Zhu, B., Wang, L., Pang, Q., Wang, S., Jiao, J., Song, D., & Jordan, M. I. (2023, April). Byzantine-robust federated learning with optimal statistical rates. In International Conference on Artificial Intelligence and Statistics (pp. 3151-3178). PMLR.
