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

## Prompt Engineering Techniques

Several prompt engineering techniques are used to optimize AI responses:

### 1. Role Prompting

```python
"You are a real estate analytics expert specializing in property clustering and lease-up analysis."
```
This establishes domain expertise for the model, improving the relevance and depth of responses.

### 2. Structured Data Presentation

```python
Cluster statistics:
{cluster_stats}
```
Presenting structured data in a standardized format helps the model interpret numerical information correctly.

### 3. Specific Output Formatting Guidelines

```python
Format your response with markdown headings and bullet points for clarity.
```
This ensures consistent, readable responses that integrate well with the Streamlit interface.

### 4. Task Decomposition

The prompt breaks down the task into specific components:
- Understanding cluster statistics
- Addressing the user's query
- Identifying patterns
- Providing actionable insights

### 5. Temperature Control

```python
temperature=0.5
```
A moderate temperature (0.5) balances creativity with factual accuracy - critical for data analysis tasks.

## Implementation Considerations

1. **Fallback System**: The dual approach allows the dashboard to function even without an API key
2. **Security**: API keys are treated as passwords and never stored in session
3. **Response Quality Control**: Parameters like temperature and max_tokens are tuned for optimal responses
4. **Error Handling**: Comprehensive error catching with informative messages

This hybrid approach creates a robust, user-friendly interface for exploring property clustering results using natural language.
