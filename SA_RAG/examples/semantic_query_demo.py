"""语义查询演示

演示 SA-RAG 的语义检索能力：
- 语义相似性匹配（而非关键词匹配）
- 同义词和概念理解
- 混合检索（向量 + BM25）
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "../python"))

from sa_rag import RAG


def main():
    print("=" * 60)
    print("语义查询演示")
    print("=" * 60)
    
    # 初始化 RAG 系统
    print("\n[1] 初始化 SA-RAG 系统...")
    rag = RAG()
    print("✓ 系统初始化完成")

    # 索引文档
    print("\n[2] 索引文档...")
    documents = [
        """语义搜索理解文本的含义，而不仅仅是关键词匹配。
        
        传统的关键词搜索只能找到包含确切词汇的文档，而语义搜索能够理解查询的意图和概念。
        这使得即使用不同的词汇表达相同的意思，也能找到相关的结果。
        """,
        
        """向量嵌入是语义搜索的核心技术。
        
        通过将文本转换为高维向量，我们可以计算文本之间的语义相似度。
        相似的文本在向量空间中距离更近，这使得语义检索成为可能。
        """,
        
        """混合检索结合了多种检索方法的优势。
        
        BM25 擅长精确的关键词匹配，而向量搜索擅长语义相似性匹配。
        通过结合两者，可以获得更好的检索效果。
        """,
    ]
    
    doc_ids = rag.index_documents(documents, generate_embeddings=True)
    print(f"✓ 已索引 {len(doc_ids)} 个文档")

    # 语义查询示例
    print("\n[3] 语义查询示例...")
    
    # 示例1：同义词查询
    print("\n[3.1] 同义词查询")
    print("-" * 60)
    query1 = "语义搜索如何理解文本的含义？"
    print(f"查询: {query1}")
    result1 = rag.ask(query1, top_k=3)
    print(f"答案: {result1['answer'][:150]}...")
    print(f"检索到的节点数: {len(result1['used_semantic_nodes'])}")
    
    # 示例2：概念查询（使用不同的词汇）
    print("\n[3.2] 概念查询（不同词汇，相同含义）")
    print("-" * 60)
    query2 = "向量嵌入技术在信息检索中的作用是什么？"
    print(f"查询: {query2}")
    result2 = rag.ask(query2, top_k=3)
    print(f"答案: {result2['answer'][:150]}...")
    
    # 示例3：意图理解
    print("\n[3.3] 意图理解查询")
    print("-" * 60)
    query3 = "如何结合不同的搜索方法来提高检索效果？"
    print(f"查询: {query3}")
    result3 = rag.ask(query3, top_k=3)
    print(f"答案: {result3['answer'][:150]}...")
    
    # 比较不同检索方法
    print("\n[4] 检索方法比较...")
    print("-" * 60)
    
    test_query = "文本相似度计算"
    print(f"测试查询: {test_query}\n")
    
    # 仅使用 BM25（通过设置 use_graph=False，主要依赖关键词）
    print("检索结果详情:")
    results = rag.search(test_query, top_k=3, use_graph=False)
    for i, res in enumerate(results, 1):
        print(f"  [{i}] 分数: {res.get('score', 0):.3f}, 来源: {res.get('source', 'unknown')}")
        print(f"      文本: {res.get('text', '')[:80]}...")
    
    # 使用混合检索（向量 + BM25 + 图）
    print("\n使用混合检索（向量 + BM25 + 图扩展）:")
    results_hybrid = rag.search(test_query, top_k=3, use_graph=True)
    for i, res in enumerate(results_hybrid, 1):
        print(f"  [{i}] 分数: {res.get('score', 0):.3f}, 来源: {res.get('source', 'unknown')}")
        print(f"      文本: {res.get('text', '')[:80]}...")

    # 语义节点分析
    print("\n[5] 语义节点分析...")
    print("-" * 60)
    doc_id = doc_ids[0]
    node_ids = rag.pipeline.engine.get_node_ids_for_doc(doc_id)
    print(f"文档 {doc_id} 的语义节点结构:")
    
    for node_id in node_ids[:5]:  # 显示前5个节点
        node_info = rag.pipeline.engine.get_node_info(node_id)
        if node_info:
            level = node_info.get('level', 0)
            text = node_info.get('text', '')[:60]
            indent = "  " * level
            print(f"{indent}[L{level}] {text}...")

    print("\n" + "=" * 60)
    print("演示完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
