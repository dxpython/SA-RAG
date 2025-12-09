"""基础 RAG 演示

演示 SA-RAG 的基本功能：
- 文档索引
- 语义检索
- 答案生成
"""

import sys
import os

# 确保可以导入本地包
sys.path.append(os.path.join(os.path.dirname(__file__), "../python"))

from sa_rag import RAG


def main():
    print("=" * 60)
    print("SA-RAG 基础演示")
    print("=" * 60)
    
    # 初始化 RAG 系统
    print("\n[1] 初始化 SA-RAG 系统...")
    rag = RAG(
        llm_provider="mock",  # 使用模拟模式（不需要 API 密钥）
        embedding_provider="mock",
    )
    print("✓ 系统初始化完成")

    # 索引文档
    print("\n[2] 索引文档...")
    documents = [
        """高血压是一种以动脉压力升高为特征的慢性疾病。
        
        诊断标准：
        1. 收缩压 ≥ 140 mmHg
        2. 舒张压 ≥ 90 mmHg
        3. 或两者同时满足
        
        高血压分为原发性和继发性两种类型。原发性高血压占90%以上，病因不明。
        继发性高血压由其他疾病引起，如肾脏疾病、内分泌疾病等。
        
        治疗原则：
        - 生活方式干预：低盐饮食、适量运动、戒烟限酒
        - 药物治疗：ACE抑制剂、ARB、利尿剂等
        """,
        
        """糖尿病是一种以高血糖为特征的代谢性疾病。
        
        诊断标准：
        1. 空腹血糖 ≥ 7.0 mmol/L
        2. 随机血糖 ≥ 11.1 mmol/L
        3. 糖化血红蛋白 ≥ 6.5%
        
        糖尿病分为1型和2型。1型糖尿病是自身免疫性疾病，需要胰岛素治疗。
        2型糖尿病与胰岛素抵抗有关，可通过口服药物和生活方式干预控制。
        """,
        
        """心血管疾病是心脏和血管系统的疾病总称。
        
        常见类型：
        - 冠心病：冠状动脉狭窄或阻塞
        - 心力衰竭：心脏泵血功能下降
        - 心律失常：心跳节律异常
        
        危险因素包括高血压、高血脂、糖尿病、吸烟、肥胖等。
        预防措施：控制危险因素、健康饮食、规律运动。
        """,
    ]
    
    doc_ids = rag.index_documents(documents, generate_embeddings=True)
    print(f"✓ 已索引 {len(doc_ids)} 个文档")

    # 问答示例
    queries = [
        "高血压的诊断标准是什么？",
        "糖尿病的治疗方法有哪些？",
        "心血管疾病的危险因素是什么？",
    ]
    
    for i, query in enumerate(queries, 1):
        print(f"\n[3.{i}] 查询: {query}")
        print("-" * 60)
        
        result = rag.ask(
            query=query,
            top_k=3,
            use_graph=True,
            use_memory=False,
        )
        
        print(f"\n答案:")
        print(result['answer'])
        
        print(f"\n使用的语义节点数量: {len(result['used_semantic_nodes'])}")
        print(f"使用的图节点数量: {len(result['used_graph_nodes'])}")
        print(f"总检索结果数: {result['total_results']}")
        
        print(f"\n评分详情:")
        for node_id, details in list(result['scoring_details'].items())[:3]:
            print(f"  节点 {node_id}: 分数={details['score']:.3f}, 来源={details['source']}")

    # 长期记忆示例
    print("\n[4] 长期记忆功能演示...")
    rag.add_memory("用户偏好：喜欢详细的医学解释", importance=0.8)
    rag.add_memory("用户关注：心血管疾病的预防", importance=0.9)
    
    memory_query = "我之前关注过什么疾病？"
    print(f"查询: {memory_query}")
    memory_result = rag.ask(memory_query, use_memory=True)
    print(f"答案: {memory_result['answer']}")

    print("\n" + "=" * 60)
    print("演示完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
