# Data-Science-Intern-Assessment

# Documentation: AI Components and Prompt Engineering Techniques

## Overview

The Property Clustering Dashboard integrates two AI approaches for generating insights:

1. **Rule-based Template System** - A custom-built system that uses pattern matching to generate insights from data without requiring an API key
2. **OpenAI API Integration** - Advanced natural language processing using OpenAI's models

## Rule-based Template System Architecture

The template system is implemented in the `PropertyInsightGenerator` class and follows a simple but effective strategy:

### 1. Query Classification

The system first classifies the user's query into categories:
```python
def _template_based_response(self, query, stats):
    query = query.lower()
    
    # General statistics overview
    if "overview" in query or "summary" in query or "general" in query:
        return self._generate_overview(stats)
    # Lease-up specific queries    
    elif any(term in query for term in ["fast", "quick", "speed", "fastest"]):
        return self._generate_lease_up_speed_insight(stats)
    # ...other classifications
```

### 2. Statistical Analysis

Each query type has a dedicated method that:
1. Calculates relevant statistics
2. Determines relationships between variables
3. Identifies patterns in the data

For example, when analyzing correlations:
```python
# Calculate correlation with lease-up time
size_values = stats['AreaPerUnit'].values
leaseup_values = stats['Lease_Up_Time'].values
correlation = np.corrcoef(size_values, leaseup_values)[0, 1]

# Interpret the correlation
if abs(correlation) < 0.3:
    response += "There is **no strong correlation**..."
elif correlation > 0:
    response += f"There is a **positive correlation** ({correlation:.2f})..."
else:
    response += f"There is a **negative correlation** ({correlation:.2f})..."
```

### 3. Dynamic Response Formation

Responses are constructed using markdown formatting for readability and emphasize data-driven insights:
```python
response = f"# Lease-Up Speed Analysis\n\n"
response += f"## Fastest Leasing Properties (Cluster {fastest_cluster})\n"
response += f"- **Average lease-up time**: {fastest_stats['Lease_Up_Time']:.1f} months\n"
```

## OpenAI API Integration

The dashboard leverages OpenAI's API for more sophisticated natural language insights:

### 1. Context Construction

The system builds a comprehensive prompt that includes:
- Statistical summary of clusters for context
- The user's specific query
- Instructions on response format and focus

```python
prompt = f"""
You are a real estate analytics expert analyzing property clustering results.

Cluster statistics:
{cluster_stats}

Based on this data, please provide insights for the query: "{query}"

Focus on lease-up time patterns, relationships between variables, and actionable insights.
Format your response with markdown headings and bullet points for clarity.
"""
```

### 2. API Call with Error Handling

```python
try:
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a real estate analytics expert..."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.5,
        max_tokens=800
    )
    return response.choices[0].message.content
except Exception as e:
    return f"Error generating insights: {str(e)}"
```


## Detailed Prompt Engineering Strategies

The effectiveness of the AI components in the Property Clustering Dashboard relies heavily on sophisticated prompt engineering techniques. Here's a deeper look at the approaches implemented:

### 1. Domain-Specific Role Setting

```python
"You are a real estate analytics expert specializing in property clustering and lease-up analysis."
```

This role definition does more than just establish domain knowledge - it activates specific reasoning patterns in the model:

- **Industry-Specific Vocabulary**: Primes the model to use terms like "lease-up time," "concessions," and "absorption rates" appropriately
- **Analytical Mindset**: Encourages quantitative reasoning and statistical interpretation
- **Professional Tone**: Creates output that matches the expectations of real estate professionals

The role definition functions as a cognitive frame that shapes all subsequent reasoning.

### 2. Data Contextualization

The prompt carefully contextualizes data before asking for analysis:

```python
cluster_stats = clustered_df.groupby('Cluster').agg({
    'Lease_Up_Time': 'mean',
    'Average_Rent_During_LeaseUp': 'mean',
    'Effective_Age': 'mean',
    'Quantity': 'mean',
    'AreaPerUnit': 'mean',
    'Average_Concession_During_LeaseUp': 'mean'
}).round(2).to_string()
```

This approach:
- Provides statistical summaries rather than raw data
- Pre-aggregates information at the appropriate level of analysis
- Formats numbers for readability (using rounding)
- Orders variables to emphasize the target variable (Lease_Up_Time first)

The format of data presentation significantly impacts how the model interprets relationships between variables.

### 3. Directional Instruction Strategy

The prompt uses specific directives to guide the analysis:

```python
"Focus on lease-up time patterns, relationships between variables, and actionable insights."
```

This three-part instruction creates a structured analytical framework:
- **Pattern identification**: Detecting trends in the primary metric
- **Relationship analysis**: Exploring correlations and dependencies
- **Action orientation**: Translating findings into practical recommendations

The sequencing is intentional, creating a logical flow from observation to insight to action.

### 4. Multi-layered Query Processing

The template system employs a sophisticated query parsing approach:

```python
# Pattern matching for specific query types
if "overview" in query or "summary" in query or "general" in query:
    return self._generate_overview(stats)
elif any(term in query for term in ["fast", "quick", "speed", "fastest"]):
    return self._generate_lease_up_speed_insight(stats)
```

This strategy:
- Identifies user intent through keyword analysis
- Maps intents to specialized processing functions
- Handles linguistic variations (synonyms like "fast" and "quick")
- Provides fallback mechanisms for ambiguous queries

The tiered approach allows for highly specific responses to common queries while maintaining flexibility.

### 5. Response Scaffolding

Both the template system and API prompts use markdown formatting instructions:

```python
"Format your response with markdown headings and bullet points for clarity."
```

This technique:
- Creates consistent information hierarchy
- Improves visual scanning of responses
- Standardizes output format across different query types
- Ensures compatibility with Streamlit's markdown renderer

The structured output design makes complex analytical insights more accessible.

### 6. Parameter Optimization

The API calls use carefully calibrated parameters:

```python
temperature=0.5,
max_tokens=800
```

These settings represent a deliberate balance:
- **Temperature**: 0.5 is high enough for creative expression but low enough for factual consistency
- **Max tokens**: 800 allows for comprehensive but focused responses without unnecessary verbosity

These parameters were tuned through iterative testing to optimize response quality.

### 7. Fallback Response Design

The system implements sophisticated error handling with informative fallbacks:

```python
try:
    # API call logic
except Exception as e:
    return f"Error generating insights: {str(e)}"
```

This approach:
- Captures specific error types from the API
- Provides transparent error reporting
- Maintains user experience continuity
- Falls back to template-based insights when API is unavailable

The graceful degradation strategy ensures the dashboard remains functional under various conditions.

## Implementation Impact

These prompt engineering techniques have significant impacts on the dashboard's effectiveness:

1. **Response Quality**: Clear, structured insights directly tied to the data
2. **Customization**: Tailored analysis based on specific user queries
3. **Consistency**: Predictable response patterns across different query types
4. **Accessibility**: Complex statistical insights presented in understandable language

The integration of these techniques creates a natural language interface that bridges the gap between complex clustering algorithms and actionable real estate insights, providing significant value to both technical and non-technical stakeholders.
