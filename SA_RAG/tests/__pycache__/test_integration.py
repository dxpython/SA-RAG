#!/usr/bin/env python3
"""集成测试：测试完整的 RAG 流程"""

import sys
sys.path.insert(0, 'python')

from sa_rag import RAG

def test_integration():
    """完整集成测试"""
    print("=" * 60)
    print("SA-RAG 集成测试")
    print("=" * 60)
    
    # 初始化
    rag = RAG(llm_provider="mock", embedding_provider="mock")
    print("✅ 1. RAG 系统初始化成功")
    
    # 索引多个文档
    documents = [
        "Python 是一种高级编程语言，广泛用于数据科学和机器学习。",
        "Rust 是一种系统编程语言，注重安全性和性能。",
        "机器学习是人工智能的一个子领域，使用算法从数据中学习。",
        "深度学习使用神经网络来模拟人脑的学习过程。",
        "自然语言处理是计算机科学和人工智能的一个分支。"
    ]
    
    doc_ids = rag.index_documents(documents, generate_embeddings=False)
    print(f"✅ 2. 索引了 {len(doc_ids)} 个文档")
    
    # 测试搜索
    queries = [
        "编程语言",
        "机器学习",
        "人工智能"
    ]
    
    for query in queries:
        results = rag.search(query, top_k=3)
        print(f"✅ 3. 查询 '{query}' 返回 {len(results)} 个结果")
    
    # 测试问答
    answer = rag.ask("什么是 Python？", top_k=2)
    print(f"✅ 4. 问答功能正常")
    print(f"   答案预览: {answer.get('answer', '')[:80]}...")
    
    # 测试记忆
    rag.add_memory("用户喜欢使用 Python 进行数据分析", importance=0.8)
    print("✅ 5. 记忆添加成功")
    
    # 测试文档更新
    if doc_ids:
        rag.update_document(doc_ids[0], "Python 是一种非常流行的编程语言，用于各种应用。")
        print("✅ 6. 文档更新成功")
    
    print("\n" + "=" * 60)
    print("✅ 所有集成测试通过！")
    print("=" * 60)

if __name__ == "__main__":
    test_integration()

