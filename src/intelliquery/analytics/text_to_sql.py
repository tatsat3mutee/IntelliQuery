"""
Text-to-SQL Agent - Convert Natural Language to SQL Queries
============================================================
Enables natural language queries against the telco_churn_data table.

Features:
- Dynamic schema detection
- Natural language to SQL conversion
- Query validation and sanitization
- Result formatting for display
"""

import logging
import re
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum

from ..core.database import db_client
from ..core.config import config

logger = logging.getLogger(__name__)


class AggregationType(Enum):
    """Types of aggregations we support"""
    COUNT = "COUNT"
    SUM = "SUM"
    AVG = "AVG"
    MIN = "MIN"
    MAX = "MAX"
    NONE = "NONE"


@dataclass
class SQLQuery:
    """Structured SQL query components"""
    select_columns: List[str]
    aggregation: Optional[AggregationType] = None
    agg_column: Optional[str] = None
    where_conditions: List[str] = None
    group_by: Optional[str] = None
    order_by: Optional[str] = None
    order_dir: str = "DESC"
    limit: int = 100
    
    def __post_init__(self):
        if self.where_conditions is None:
            self.where_conditions = []


class TextToSQLAgent:
    """Agent for converting natural language to SQL queries"""
    
    def __init__(self):
        self._schema_cache: Dict[str, str] = {}
        self._column_aliases: Dict[str, str] = {}
        self._churn_column: Optional[str] = None
        
    def refresh_schema(self) -> Dict[str, str]:
        """Fetch and cache the table schema"""
        try:
            result = db_client.query(f"DESCRIBE TABLE {config.CHURN_TABLE}")
            if result:
                self._schema_cache = {}
                for row in result:
                    col_name = row.get('col_name', row.get('column_name', ''))
                    data_type = row.get('data_type', row.get('type', ''))
                    if col_name and not col_name.startswith('#'):
                        self._schema_cache[col_name.lower()] = data_type.upper()
                
                # Build column aliases for natural language mapping
                self._build_column_aliases()
                
                logger.info(f"Schema refreshed: {len(self._schema_cache)} columns")
            return self._schema_cache
        except Exception as e:
            logger.error(f"Failed to refresh schema: {e}")
            return {}
    
    def _build_column_aliases(self):
        """Build natural language aliases for columns"""
        self._column_aliases = {}
        
        # Common mappings
        alias_patterns = {
            'customer': ['customer', 'customerid', 'customer_id', 'cust_id'],
            'gender': ['gender', 'sex'],
            'senior': ['senior', 'senior_citizen', 'seniorcitizen', 'is_senior'],
            'tenure': ['tenure', 'tenure_months', 'months', 'tenuremonths'],
            'contract': ['contract', 'contract_type', 'contracttype'],
            'monthly': ['monthly', 'monthly_charges', 'monthlycharges', 'monthly_charge'],
            'total': ['total', 'total_charges', 'totalcharges', 'total_charge', 'revenue'],
            'churn': ['churn', 'churn_value', 'churnvalue', 'churned', 'is_churn'],
            'internet': ['internet', 'internet_service', 'internetservice'],
            'payment': ['payment', 'payment_method', 'paymentmethod', 'payment_type'],
            'phone': ['phone', 'phone_service', 'phoneservice'],
            'partner': ['partner', 'has_partner'],
            'dependents': ['dependents', 'has_dependents'],
        }
        
        for natural_name, patterns in alias_patterns.items():
            for pattern in patterns:
                for col in self._schema_cache.keys():
                    if pattern in col.lower():
                        self._column_aliases[natural_name] = col
                        if natural_name == 'churn':
                            self._churn_column = col
                        break
    
    def get_column_for_term(self, term: str) -> Optional[str]:
        """Find the actual column name for a natural language term"""
        term_lower = term.lower().strip()
        
        # Direct match
        if term_lower in self._schema_cache:
            return term_lower
        
        # Check aliases
        for alias, col in self._column_aliases.items():
            if alias in term_lower or term_lower in alias:
                return col
        
        # Fuzzy match against column names
        for col in self._schema_cache.keys():
            if term_lower in col or col in term_lower:
                return col
        
        return None
    
    def parse_question(self, question: str) -> SQLQuery:
        """Parse natural language question into SQL components"""
        question_lower = question.lower().strip()
        
        # Default query structure
        query = SQLQuery(select_columns=['*'])
        
        # Detect aggregation type
        if any(word in question_lower for word in ['how many', 'count', 'number of', 'total number']):
            query.aggregation = AggregationType.COUNT
        elif any(word in question_lower for word in ['average', 'avg', 'mean']):
            query.aggregation = AggregationType.AVG
        elif any(word in question_lower for word in ['total', 'sum', 'sum of']):
            query.aggregation = AggregationType.SUM
        elif any(word in question_lower for word in ['maximum', 'max', 'highest', 'largest']):
            query.aggregation = AggregationType.MAX
        elif any(word in question_lower for word in ['minimum', 'min', 'lowest', 'smallest']):
            query.aggregation = AggregationType.MIN
        
        # Detect aggregation column
        if query.aggregation and query.aggregation != AggregationType.COUNT:
            for term in ['charge', 'revenue', 'cost', 'monthly', 'total', 'tenure', 'months']:
                col = self.get_column_for_term(term)
                if col and question_lower.find(term) != -1:
                    query.agg_column = col
                    break
        
        # Detect WHERE conditions
        # Churn filter
        if any(word in question_lower for word in ['churned', 'who churned', 'churn = 1', 'left', 'departed']):
            if self._churn_column:
                query.where_conditions.append(f"{self._churn_column} = 1")
        elif any(word in question_lower for word in ['not churned', 'stayed', 'retained', 'active']):
            if self._churn_column:
                query.where_conditions.append(f"{self._churn_column} = 0")
        
        # Contract type filter
        if 'month-to-month' in question_lower or 'monthly contract' in question_lower:
            contract_col = self.get_column_for_term('contract')
            if contract_col:
                query.where_conditions.append(f"{contract_col} = 'Month-to-month'")
        elif 'one year' in question_lower or 'yearly' in question_lower:
            contract_col = self.get_column_for_term('contract')
            if contract_col:
                query.where_conditions.append(f"{contract_col} = 'One year'")
        elif 'two year' in question_lower:
            contract_col = self.get_column_for_term('contract')
            if contract_col:
                query.where_conditions.append(f"{contract_col} = 'Two year'")
        
        # Gender filter
        if 'male' in question_lower and 'female' not in question_lower:
            gender_col = self.get_column_for_term('gender')
            if gender_col:
                query.where_conditions.append(f"{gender_col} = 'Male'")
        elif 'female' in question_lower:
            gender_col = self.get_column_for_term('gender')
            if gender_col:
                query.where_conditions.append(f"{gender_col} = 'Female'")
        
        # Senior citizen filter
        if 'senior' in question_lower:
            senior_col = self.get_column_for_term('senior')
            if senior_col:
                query.where_conditions.append(f"{senior_col} = 1")
        
        # Internet service filter
        if 'fiber' in question_lower:
            internet_col = self.get_column_for_term('internet')
            if internet_col:
                query.where_conditions.append(f"{internet_col} = 'Fiber optic'")
        elif 'dsl' in question_lower:
            internet_col = self.get_column_for_term('internet')
            if internet_col:
                query.where_conditions.append(f"{internet_col} = 'DSL'")
        
        # Detect GROUP BY
        if any(word in question_lower for word in ['by contract', 'per contract', 'for each contract']):
            contract_col = self.get_column_for_term('contract')
            if contract_col:
                query.group_by = contract_col
        elif any(word in question_lower for word in ['by gender', 'per gender', 'for each gender']):
            gender_col = self.get_column_for_term('gender')
            if gender_col:
                query.group_by = gender_col
        elif any(word in question_lower for word in ['by internet', 'per internet']):
            internet_col = self.get_column_for_term('internet')
            if internet_col:
                query.group_by = internet_col
        elif any(word in question_lower for word in ['by payment', 'per payment']):
            payment_col = self.get_column_for_term('payment')
            if payment_col:
                query.group_by = payment_col
        
        # Detect ORDER BY and LIMIT
        if any(word in question_lower for word in ['top', 'highest', 'most']):
            query.order_dir = 'DESC'
            # Extract limit number
            numbers = re.findall(r'\b(\d+)\b', question_lower)
            if numbers:
                query.limit = min(int(numbers[0]), 1000)
            else:
                query.limit = 10
        elif any(word in question_lower for word in ['bottom', 'lowest', 'least']):
            query.order_dir = 'ASC'
            numbers = re.findall(r'\b(\d+)\b', question_lower)
            if numbers:
                query.limit = min(int(numbers[0]), 1000)
            else:
                query.limit = 10
        
        # Detect order column for top/bottom queries
        if 'top' in question_lower or 'bottom' in question_lower or 'highest' in question_lower or 'lowest' in question_lower:
            for term in ['charge', 'revenue', 'cost', 'monthly', 'total', 'tenure']:
                if term in question_lower:
                    col = self.get_column_for_term(term)
                    if col:
                        query.order_by = col
                        break
            if not query.order_by:
                # Default to total charges
                query.order_by = self.get_column_for_term('total') or self.get_column_for_term('monthly')
        
        # List queries
        if any(word in question_lower for word in ['list', 'show', 'display', 'get all']):
            query.aggregation = AggregationType.NONE
            # Limit list queries
            if 'all' not in question_lower:
                query.limit = 50
        
        return query
    
    def build_sql(self, query: SQLQuery) -> str:
        """Build SQL statement from parsed query"""
        parts = []
        
        # SELECT clause
        if query.aggregation == AggregationType.COUNT:
            if query.group_by:
                parts.append(f"SELECT {query.group_by}, COUNT(*) as count")
            else:
                parts.append("SELECT COUNT(*) as count")
        elif query.aggregation in (AggregationType.SUM, AggregationType.AVG, AggregationType.MIN, AggregationType.MAX):
            agg_col = query.agg_column or self.get_column_for_term('monthly') or '*'
            agg_name = query.aggregation.value
            if query.group_by:
                parts.append(f"SELECT {query.group_by}, ROUND({agg_name}({agg_col}), 2) as {agg_name.lower()}_{agg_col}")
            else:
                parts.append(f"SELECT ROUND({agg_name}({agg_col}), 2) as {agg_name.lower()}_{agg_col}")
        else:
            # Non-aggregation query - select useful columns
            select_cols = query.select_columns
            if select_cols == ['*']:
                # Select a sensible subset
                useful_cols = []
                for term in ['customer', 'gender', 'tenure', 'contract', 'monthly', 'total', 'churn']:
                    col = self.get_column_for_term(term)
                    if col:
                        useful_cols.append(col)
                if useful_cols:
                    select_cols = useful_cols
            parts.append(f"SELECT {', '.join(select_cols)}")
        
        # FROM clause
        parts.append(f"FROM {config.CHURN_TABLE}")
        
        # WHERE clause
        if query.where_conditions:
            parts.append(f"WHERE {' AND '.join(query.where_conditions)}")
        
        # GROUP BY clause
        if query.group_by and query.aggregation:
            parts.append(f"GROUP BY {query.group_by}")
            # Add ORDER BY for group results
            parts.append(f"ORDER BY count DESC" if query.aggregation == AggregationType.COUNT else f"ORDER BY 2 DESC")
        
        # ORDER BY clause (for non-group queries)
        if query.order_by and not query.group_by:
            parts.append(f"ORDER BY {query.order_by} {query.order_dir}")
        
        # LIMIT clause
        if query.limit:
            parts.append(f"LIMIT {query.limit}")
        
        return " ".join(parts)
    
    def execute_query(self, question: str) -> Dict[str, Any]:
        """
        Main entry point: Convert question to SQL and execute
        
        Returns:
            {
                "success": bool,
                "sql": str,           # Generated SQL
                "results": list,      # Query results
                "answer": str,        # Natural language answer
                "row_count": int,
                "columns": list,
                "error": str          # If failed
            }
        """
        try:
            # Ensure schema is loaded
            if not self._schema_cache:
                self.refresh_schema()
            
            if not self._schema_cache:
                return {
                    "success": False,
                    "error": "No churn data table found. Please upload churn data first."
                }
            
            # Parse question
            query = self.parse_question(question)
            
            # Build SQL
            sql = self.build_sql(query)
            logger.info(f"Generated SQL: {sql}")
            
            # Execute query
            results = db_client.query(sql)
            
            if results is None:
                results = []
            
            # Format results for JSON
            formatted_results = []
            for row in results:
                formatted_row = {}
                for k, v in row.items():
                    # Handle special types
                    if hasattr(v, 'isoformat'):
                        formatted_row[k] = str(v)
                    elif isinstance(v, (int, float)):
                        formatted_row[k] = v
                    else:
                        formatted_row[k] = str(v) if v is not None else None
                formatted_results.append(formatted_row)
            
            # Generate natural language answer
            answer = self._format_answer(question, query, formatted_results)
            
            # Get column names
            columns = list(results[0].keys()) if results else []
            
            return {
                "success": True,
                "sql": sql,
                "results": formatted_results[:100],  # Limit returned results
                "answer": answer,
                "row_count": len(results),
                "columns": columns
            }
            
        except Exception as e:
            logger.error(f"Text-to-SQL error: {e}")
            return {
                "success": False,
                "error": str(e),
                "sql": sql if 'sql' in locals() else None
            }
    
    def _format_answer(self, question: str, query: SQLQuery, results: List[Dict]) -> str:
        """Format query results as natural language"""
        if not results:
            return "No data found matching your criteria."
        
        question_lower = question.lower()
        
        # Count queries
        if query.aggregation == AggregationType.COUNT:
            if query.group_by:
                # Grouped count
                groups = [f"{r.get(query.group_by, 'Unknown')}: {r.get('count', 0):,}" for r in results[:10]]
                return f"Here's the breakdown:\n" + "\n".join(f"• {g}" for g in groups)
            else:
                count = results[0].get('count', 0)
                
                # Add context based on filters
                context = ""
                if 'churned' in question_lower:
                    context = " churned customers"
                elif 'customer' in question_lower:
                    context = " customers"
                
                return f"There are **{count:,}**{context} in the database."
        
        # Aggregation queries
        if query.aggregation in (AggregationType.AVG, AggregationType.SUM, AggregationType.MIN, AggregationType.MAX):
            if query.group_by:
                groups = []
                for r in results[:10]:
                    val_key = [k for k in r.keys() if k != query.group_by][0]
                    groups.append(f"{r.get(query.group_by, 'Unknown')}: ${r.get(val_key, 0):,.2f}" if 'charge' in val_key else f"{r.get(query.group_by, 'Unknown')}: {r.get(val_key, 0):,.2f}")
                return f"Here's the breakdown:\n" + "\n".join(f"• {g}" for g in groups)
            else:
                value = list(results[0].values())[0]
                agg_name = query.aggregation.name.lower()
                col_name = query.agg_column or "value"
                
                # Format based on column type
                if isinstance(value, (int, float)):
                    if 'charge' in col_name.lower() or 'revenue' in col_name.lower() or 'cost' in col_name.lower():
                        return f"The {agg_name} {col_name.replace('_', ' ')} is **${value:,.2f}**"
                    elif 'tenure' in col_name.lower() or 'month' in col_name.lower():
                        return f"The {agg_name} {col_name.replace('_', ' ')} is **{value:,.1f} months**"
                    else:
                        return f"The {agg_name} {col_name.replace('_', ' ')} is **{value:,.2f}**"
                return f"The {agg_name} is {value}"
        
        # List queries
        if len(results) > 1:
            # Return summary for list
            return f"Found **{len(results)}** records. Showing the data in the table below."
        
        # Single result
        if len(results) == 1:
            r = results[0]
            items = [f"{k.replace('_', ' ').title()}: {v}" for k, v in r.items() if v is not None]
            return "Result:\n" + "\n".join(f"• {item}" for item in items[:10])
        
        return "Query executed successfully."
    
    def get_schema_summary(self) -> Dict[str, Any]:
        """Get a summary of available columns for UI display"""
        if not self._schema_cache:
            self.refresh_schema()
        
        return {
            "columns": list(self._schema_cache.keys()),
            "column_types": self._schema_cache,
            "aliases": self._column_aliases,
            "churn_column": self._churn_column
        }


# Global agent instance
text_to_sql_agent = TextToSQLAgent()
