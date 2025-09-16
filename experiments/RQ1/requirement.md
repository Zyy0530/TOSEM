这是我论文的部分：

\textbf{Ground-Truth Dataset Construction.} We construct a ground-truth dataset to evaluate
detector effectiveness through two complementary approaches. (i) For factory contracts, we
utilize Google BigQuery's Ethereum \textit{traces} table~\cite{bigquery-ethereum-traces} to identify
contracts that executed CREATE or CREATE2 operations in non-constructor contexts with successful
status, ensuring definitive factory classification through on-chain execution records. (ii) For non-factory
contracts, we obtain verified smart contracts from Etherscan~\cite{etherscan-verified-contracts}
and filter those whose source code contains no ``new'', ``create'', or ``create2'' keywords, guaranteeing
non-factory classification. This methodology yields \textcolor{red}{xx} factory contracts with unique
bytecode and \textcolor{red}{xx} non-factory contracts with unique bytecode for evaluation.


需求：你需要编写一个python脚本，用于度量factory_detector(阅读factory_detector.py)的三个方面
- (1)precision - (2)recall - (3)执行时间
为此，你需要编写一个测试脚本，用于测试这些指标，具体来说：
# 数据集的构建：
首先，你需要在Google BigQuery上创建一张远程表，GroundTruthDataset，这个表包含下面的字段：
- address:str
- bytecode:str
- is_factory_ground_truth:bool
- is_factory_detected:bool
- execution_time:int
(1)export-verified-contractaddress-opensource-license.csv给出了所有Eth上被验证的合约的地址，你需要编写函数：
根据地址下载合约源码；通过字符串匹配的方式，过滤包含“create” / “create2” / “new”关键字的合约，得到的合约一定不是Factory Contract
从而得到Non-factory contract，对于每一个合约，你需要先调用Etherscan API获取源码, 如果确认不是工厂合约（不包含上述关键字），然后获取bytecode
获取bytecode后，将其insert到GroundTruthDataset中，在insert过程中，需要注意，先检查bytecode是否是unique的，如果已有，则不insert这个记录
(2)你需要从trace表获取factory contract（注意，构造函数中进行create操作的合约不是工厂合约，你在编写sql的时候需要注意这一点，排除这类合约）
类似的，你判断它是一个工厂合约后，你需要在得到合约bytecode后，将其insert到GroundTruthDataset中，在insert过程中，需要注意，先检查bytecode是否是unique的，如果已有，则不insert这个记录

请你先帮我构造这个数据集，然后我们再展开测试实验，将脚本编写在ground_truth_dataset.py中


# 最终，根据precision和recall的计算公式，计算precision和recall的值

