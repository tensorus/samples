// tensorus-demo-page/src/demoData.js
export const demos = [
  {
    id: 'financial-news-impact',
    title: 'Financial News Impact Demo',
    shortDescription: 'Illustrates using multi-modal context for predicting financial news impact with RAG.',
    longDescription: 'This demo showcases how Tensorus can provide richer, multi-modal context for AI models to predict the impact of financial news on a network of assets. It contrasts the Tensorus approach (using multiple, structured tensors per news event) with traditional vector database approaches (using a single embedding per event) for Retrieval Augmented Generation (RAG). The goal is to show how more comprehensive data can lead to better informed inputs for hypothetical downstream predictive models.',
    thumbnailUrl: 'https://placehold.co/600x360/007AFF/FFFFFF/png?text=Financial+Demo',
    visualsPath: 'https://placehold.co/800x450/007AFF/FFFFFF/png?text=Financial+News+Impact+Detail', // Placeholder for actual screenshot of the demo page
    keyFeatures: [
      'Storing multi-modal data (news text, market features, sentiment scores) as distinct, yet related, tensors.',
      'Enabling Retrieval Augmented Generation (RAG) with rich, multi-faceted tensor context.',
      'Representing textual nuances and quantitative data in a structured tensor format.',
      'Facilitating more informed inputs for hypothetical downstream predictive models.'
    ],
    readmeLink: 'https://github.com/GoogleCloudPlatform/tensorus/blob/main/README_financial_news_impact_demo.md',
    tags: ['Finance', 'RAG', 'Multi-modal', 'NLP'],
    localPort: 8501, // Default Streamlit port
    streamlitCommand: 'streamlit run financial_news_impact_demo.py'
  },
  {
    id: 'smart-story-analyzer',
    title: 'Smart Story Analyzer Demo',
    shortDescription: 'Analyzes character relationships and sentiment evolution in literary texts using tensor representations.',
    longDescription: 'This demo highlights how Tensorus can be used to analyze character relationships and sentiment evolution within literary texts. It processes story snippets, stores various analytical representations as tensors (like sentence embeddings, character-specific sentiment flows, and interaction matrices), and allows users to explore how character dynamics change throughout the narrative. It emphasizes storing and retrieving complex relational data that AI agents can easily process for insights.',
    thumbnailUrl: 'https://placehold.co/600x360/34C759/FFFFFF/png?text=Story+Analyzer',
    visualsPath: 'https://placehold.co/800x450/34C759/FFFFFF/png?text=Smart+Story+Analyzer+Detail', // Placeholder for actual screenshot of the demo page
    keyFeatures: [
      'Transforms textual narratives into multiple structured tensors.',
      'Analyzing the temporal evolution of relationships and sentiments by operating on sequences of tensors.',
      'Enabling more nuanced queries about character dynamics beyond simple keyword searches.',
      'Storing and retrieving complex relational data for AI agents.'
    ],
    readmeLink: 'https://github.com/GoogleCloudPlatform/tensorus/blob/main/README_story_analyzer_demo.md',
    tags: ['NLP', 'Literature', 'Sentiment Analysis', 'Relationships'],
    localPort: 8501, // Default Streamlit port (Note: if running both simultaneously, one will take another port like 8502)
    streamlitCommand: 'streamlit run story_analyzer_demo.py'
  }
  // Future demos will be added here
];
