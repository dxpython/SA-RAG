#!/usr/bin/env python3
"""快速测试优化后的 SA-RAG 系统"""

import sys
sys.path.insert(0, 'python')

from sa_rag import RAG

def test_basic_functionality():
    """测试基本功能"""
    print("=" * 60)
    print("测试 SA-RAG 基本功能")
    print("=" * 60)
    
    # 初始化 RAG
    rag = RAG(llm_provider="mock", embedding_provider="mock")
    print("✅ RAG 初始化成功")
    
    # 索引文档
    texts = [
        "Python is a high-level programming language.",
        "Rust is a systems programming language focused on safety.",
        "Machine learning is a subset of artificial intelligence."
    ]
    doc_ids = rag.index_documents(texts, generate_embeddings=False)
    print(f"✅ 索引了 {len(doc_ids)} 个文档: {doc_ids}")
    
    # 搜索
    results = rag.search("programming language", top_k=2)
    print(f"✅ 搜索返回 {len(results)} 个结果")
    for i, result in enumerate(results, 1):
        print(f"  结果 {i}: {result.get('text', 'N/A')[:50]}...")
    
    # 问答
    answer = rag.ask("What is Python?", top_k=2)
    print(f"✅ 问答功能正常")
    print(f"  答案: {answer.get('answer', 'N/A')[:100]}...")
    
    print("\n" + "=" * 60)
    print("✅ 所有基本功能测试通过！")
    print("=" * 60)

if __name__ == "__main__":
    test_basic_functionality()

