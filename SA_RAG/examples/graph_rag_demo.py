"""图 RAG 演示

演示 SA-RAG 的图结构检索功能：
- 语义节点的层次结构
- 图扩展检索
- 节点关系查询
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "../python"))

from sa_rag import RAG


def main():
    print("=" * 60)
    print("图 RAG 演示")
    print("=" * 60)
    
    # 初始化 RAG 系统
    print("\n[1] 初始化 SA-RAG 系统...")
    rag = RAG()
    print("✓ 系统初始化完成")

    # 索引具有层次结构的文档
    print("\n[2] 索引具有层次结构的文档...")
    documents = [
        """# 人工智能概述
        
        人工智能（AI）是计算机科学的一个分支，致力于创建能够执行通常需要人类智能的任务的系统。
        
        ## 机器学习
        
        机器学习是人工智能的一个子领域，使计算机能够从数据中学习，而无需明确编程。
        
        ### 监督学习
        
        监督学习使用标记的训练数据来学习输入和输出之间的映射关系。
        
        ### 无监督学习
        
        无监督学习从未标记的数据中发现隐藏的模式和结构。
        
        ## 深度学习
        
        深度学习是机器学习的一个子集，使用多层神经网络来学习数据的表示。
        
        ### 神经网络
        
        神经网络是受生物神经元启发的计算模型，由相互连接的节点（神经元）组成。
        
        ### 卷积神经网络
        
        卷积神经网络（CNN）特别适合处理图像数据，使用卷积层来提取特征。
        """,
        
        """# 自然语言处理
        
        自然语言处理（NLP）是人工智能的一个分支，专注于计算机与人类语言之间的交互。
        
        ## 文本分类
        
        文本分类是将文本分配到预定义类别的任务。
        
        ## 机器翻译
        
        机器翻译是使用计算机自动将文本从一种语言翻译成另一种语言。
        
        ## 情感分析
        
        情感分析是确定文本中表达的情感或观点的任务。
        """,
    ]
    
    doc_ids = rag.index_documents(documents, generate_embeddings=True)
    print(f"✓ 已索引 {len(doc_ids)} 个文档")

    # 获取图统计信息
    print("\n[3] 图结构统计...")
    stats = rag.client.get_stats()
    if "graph" in stats:
        print(f"  节点数: {stats['graph'].get('num_nodes', 0)}")
        print(f"  边数: {stats['graph'].get('num_edges', 0)}")

    # 演示图扩展检索
    print("\n[4] 图扩展检索演示...")
    queries = [
        "什么是深度学习？",
        "神经网络和卷积神经网络的关系是什么？",
        "自然语言处理有哪些应用？",
    ]
    
    for i, query in enumerate(queries, 1):
        print(f"\n[4.{i}] 查询: {query}")
        print("-" * 60)
        
        # 不使用图扩展
        result_no_graph = rag.ask(query, top_k=3, use_graph=False)
        print(f"不使用图扩展 - 检索到 {result_no_graph['total_results']} 个结果")
        
        # 使用图扩展
        result_with_graph = rag.ask(query, top_k=3, use_graph=True)
        print(f"使用图扩展 - 检索到 {result_with_graph['total_results']} 个结果")
        print(f"  图节点数: {len(result_with_graph['used_graph_nodes'])}")
        
        print(f"\n答案:")
        print(result_with_graph['answer'][:200] + "...")

    # 演示节点扩展
    print("\n[5] 节点扩展演示...")
    doc_id = doc_ids[0]
    node_ids = rag.pipeline.engine.get_node_ids_for_doc(doc_id)
    print(f"文档 {doc_id} 包含 {len(node_ids)} 个节点")
    
    if node_ids:
        # 获取第一个节点（通常是根节点）
        start_node = node_ids[0]
        
        # Try to get node info (may not be available in old versions)
        try:
            node_info = rag.pipeline.engine.get_node_info(start_node)
            if node_info:
                print(f"\n起始节点: {node_info.get('text', '')[:50]}...")
                print(f"  节点ID: {start_node}")
                print(f"  层级: {node_info.get('level', 0)}")
                print(f"  邻居数: {node_info.get('neighbor_count', 0)}")
        except AttributeError:
            # Old version doesn't have get_node_info
            print(f"\n起始节点ID: {start_node}")
            print("  (节点详细信息不可用 - 使用旧版本 Rust 模块)")
        
        # 扩展节点
        print(f"\n从节点 {start_node} 扩展（1跳）...")
        expanded = rag.pipeline.engine.expand_nodes([start_node], hops=1)
        print(f"扩展后节点数: {len(expanded)}")
        
        # 显示扩展的节点
        print("\n扩展的节点:")
        for node_id in expanded[:5]:  # 只显示前5个
            try:
                node_info = rag.pipeline.engine.get_node_info(node_id)
                if node_info:
                    text = node_info.get('text', '')[:50]
                    print(f"  - [{node_info.get('level', 0)}] {text}...")
            except AttributeError:
                # Old version - just show node ID
                print(f"  - 节点 {node_id}")

    # 智能图扩展
    print("\n[6] 智能图扩展演示...")
    if node_ids:
        seed_nodes = node_ids[:3]  # 使用前3个节点作为种子
        print(f"种子节点: {seed_nodes}")
        
        # Try smart expansion (may not be available in old versions)
        try:
            expanded_smart = rag.pipeline.engine.expand_nodes_smart(
                seed_nodes,
                hops=2,
                min_weight=0.3,
                max_nodes=20,
            )
            print(f"智能扩展后节点数: {len(expanded_smart)}")
        except AttributeError:
            # Fallback to regular expansion
            print("  (智能扩展不可用，使用常规扩展)")
            expanded_smart = rag.pipeline.engine.expand_nodes(seed_nodes, hops=2)
            print(f"常规扩展后节点数: {len(expanded_smart)}")

    print("\n" + "=" * 60)
    print("演示完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
