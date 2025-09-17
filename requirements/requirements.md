接下来，基于最新的实验结果：
- false_negatives_list.csv
- false_positives_list.csv
首先，统计FP和FN的数据（以上述CSV文件为准）
然后，阅读/Users/mac/ResearchSpace/TOSEM/requirements/temp.tex，包含 This methodology yields 548 factory contracts with unique bytecode
  and 2359 non-factory contracts with unique bytecode for evaluation.

然后计算最新的Precision和Recall， 并分别分析导致的Root Cause：
对于Recall（FN），全部是由于Proxy合约导致的，这说明Factory Detector目前的一个Limitation是无法直接分析Proxy-based Factory
对于Precision（FP）（相当少，只有4个），我们检查了合约的字节码。这是由于xxx导致的（你需要分析主要的一个原因）


在你分析后，你需要：
(1) 以科研论文的口吻，续写temp.tex，包含：Precision和Recall分别是多少，以及分别给出FN和FP的分析,最后说明，Factory Detector可以有效的xxx，从而确保后续实验数据的xxx
(2) 基于执行时间，编写一个脚本，绘制一张执行时间的CDF图，并撰写结果说明文字，用\textbf{Execution performance.}开头，说明百分之50%以及75%的合约可以在多少时间内执行完毕

这个CDF图要符合科研绘图的风格，并且要有25%、50%、75%线的绘制

