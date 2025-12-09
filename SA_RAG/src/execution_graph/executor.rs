// Graph Executor
// Executes query according to execution graph

use super::{ExecutionGraphBuilder, ExecutionNode, ExecutionEdge, QueryExecutionGraph};

pub struct GraphExecutor;

impl GraphExecutor {
    pub fn new() -> Self {
        Self
    }

    /// Build execution graph for a query
    pub fn build_execution_graph(
        &self,
        query: &str,
        intent: &str,
        knowledge_types: &[String],
    ) -> QueryExecutionGraph {
        let mut builder = ExecutionGraphBuilder::new(query.to_string());

        // Add query node
        let query_node = ExecutionNode::new(
            "query".to_string(),
            "QUERY".to_string(),
            format!("Query: {}", query),
        );
        builder.add_node(query_node);

        // Add intent node
        let intent_node = ExecutionNode::new(
            "intent".to_string(),
            "INTENT".to_string(),
            format!("Intent: {}", intent),
        );
        builder.add_node(intent_node);
        builder.add_edge("query", "intent", ExecutionEdge::new(
            "query".to_string(),
            "intent".to_string(),
            "data_flow".to_string(),
        ));

        // Add knowledge type nodes
        for (i, kt) in knowledge_types.iter().enumerate() {
            let kt_node = ExecutionNode::new(
                format!("knowledge_type_{}", i),
                "KNOWLEDGE_TYPE".to_string(),
                format!("Knowledge Type: {}", kt),
            );
            builder.add_node(kt_node.clone());
            builder.add_edge("intent", &kt_node.node_id, ExecutionEdge::new(
                "intent".to_string(),
                kt_node.node_id.clone(),
                "data_flow".to_string(),
            ));
        }

        // Add retrieval plan node
        let plan_node = ExecutionNode::new(
            "retrieval_plan".to_string(),
            "RETRIEVAL_PLAN".to_string(),
            "Retrieval Plan".to_string(),
        );
        builder.add_node(plan_node.clone());

        // Add retrieval stage nodes
        let vector_node = ExecutionNode::new(
            "vector_search".to_string(),
            "VECTOR_SEARCH".to_string(),
            "Vector Search".to_string(),
        );
        builder.add_node(vector_node.clone());
        builder.add_edge("retrieval_plan", "vector_search", ExecutionEdge::new(
            "retrieval_plan".to_string(),
            "vector_search".to_string(),
            "control_flow".to_string(),
        ));

        let bm25_node = ExecutionNode::new(
            "bm25_search".to_string(),
            "BM25_SEARCH".to_string(),
            "BM25 Search".to_string(),
        );
        builder.add_node(bm25_node.clone());
        builder.add_edge("retrieval_plan", "bm25_search", ExecutionEdge::new(
            "retrieval_plan".to_string(),
            "bm25_search".to_string(),
            "control_flow".to_string(),
        ));

        // Add fusion node
        let fusion_node = ExecutionNode::new(
            "fusion".to_string(),
            "FUSION".to_string(),
            "Result Fusion".to_string(),
        );
        builder.add_node(fusion_node.clone());
        builder.add_edge("vector_search", "fusion", ExecutionEdge::new(
            "vector_search".to_string(),
            "fusion".to_string(),
            "data_flow".to_string(),
        ));
        builder.add_edge("bm25_search", "fusion", ExecutionEdge::new(
            "bm25_search".to_string(),
            "fusion".to_string(),
            "data_flow".to_string(),
        ));

        // Add ranking node
        let ranking_node = ExecutionNode::new(
            "ranking".to_string(),
            "RANKING".to_string(),
            "Result Ranking".to_string(),
        );
        builder.add_node(ranking_node.clone());
        builder.add_edge("fusion", "ranking", ExecutionEdge::new(
            "fusion".to_string(),
            "ranking".to_string(),
            "data_flow".to_string(),
        ));

        // Add answer node
        let answer_node = ExecutionNode::new(
            "answer".to_string(),
            "ANSWER".to_string(),
            "Final Answer".to_string(),
        );
        builder.add_node(answer_node);
        builder.add_edge("ranking", "answer", ExecutionEdge::new(
            "ranking".to_string(),
            "answer".to_string(),
            "data_flow".to_string(),
        ));

        builder.build()
    }
}

impl Default for GraphExecutor {
    fn default() -> Self {
        Self::new()
    }
}

