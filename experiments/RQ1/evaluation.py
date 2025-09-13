# 需求：你需要编写一个python脚本，用于度量factory_detector(阅读factory_detector.py)的三个方面
# - (1)precision - (2)recall - (3)执行时间
# 为此，你需要编写一个测试脚本，用于测试这些指标，具体来说：
# 数据集的构建：
#  对于precision：export-verified-contractaddress-opensource-license.csv给出了所有Eth上被验证的合约的地址，你需要编写函数：
# 根据地址下载合约源码；通过字符串匹配的方式，过滤包含“create” / “create2” / “new”关键字的合约，得到的合约一定不是Factory Contract
# 使用factory_detector来运行对应的字节码（你需要筛选unique bytecode）当所有合约运行完毕后，计算：
# 多少unique bytecode的合约，以及多少合约是准确识别为非工厂合约(TN)，剩下了多少合约被不正确识别为工厂合约(FP)
#  对于recall:


# 最终，根据precision和recall的计算公式，计算precision和recall的值

# 对于每一次运行，你需要记录factory_detector的运行时间，
