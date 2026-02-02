"""
Churn Data Handler - FULLY DYNAMIC
==================================
Reads ANY CSV/Excel file, auto-detects columns, creates table, uploads data.
Zero hardcoding - works with any dataset structure.
"""

import uuid
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from io import BytesIO

import pandas as pd
import numpy as np

from ..core.database import db_client
from ..core.config import config

logger = logging.getLogger(__name__)

# SQL reserved words that need backtick escaping
SQL_RESERVED = {'count', 'order', 'group', 'select', 'from', 'where', 'join', 'left', 
                'right', 'inner', 'outer', 'on', 'and', 'or', 'not', 'null', 'true', 
                'false', 'like', 'in', 'between', 'case', 'when', 'then', 'else', 'end',
                'as', 'by', 'having', 'limit', 'offset', 'union', 'all', 'distinct',
                'index', 'key', 'primary', 'foreign', 'references', 'table', 'column',
                'create', 'drop', 'alter', 'insert', 'update', 'delete', 'values'}


def _normalize_col(col: str) -> str:
    """Normalize column name to valid SQL identifier, handle reserved words"""
    norm = str(col).lower().strip().replace(' ', '_').replace('-', '_').replace('.', '_')
    # If it's a reserved word, prefix with 'col_'
    if norm in SQL_RESERVED:
        return f"col_{norm}"
    return norm


def _infer_sql_type(series: pd.Series) -> str:
    """Infer SQL type from pandas series"""
    dtype = series.dtype
    
    if pd.api.types.is_integer_dtype(dtype):
        return 'BIGINT'
    elif pd.api.types.is_float_dtype(dtype):
        return 'DOUBLE'
    elif pd.api.types.is_bool_dtype(dtype):
        return 'BOOLEAN'
    elif pd.api.types.is_datetime64_any_dtype(dtype):
        return 'TIMESTAMP'
    else:
        # Check if it looks like a number stored as string
        try:
            non_null = series.dropna()
            if len(non_null) > 0:
                sample = non_null.head(100)
                numeric = pd.to_numeric(sample, errors='coerce')
                if numeric.notna().sum() > len(sample) * 0.8:  # 80% numeric
                    if (numeric % 1 == 0).all():
                        return 'BIGINT'
                    return 'DOUBLE'
        except:
            pass
        return 'STRING'


def _escape_sql(val: Any) -> str:
    """Escape value for SQL"""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return ''
    return str(val).replace("'", "''").replace("\\", "\\\\")


def _format_value(val: Any, sql_type: str) -> str:
    """Format value for SQL insertion based on type"""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return 'NULL'
    
    if sql_type in ('BIGINT', 'INT', 'INTEGER'):
        try:
            return str(int(float(val)))
        except:
            return 'NULL'
    elif sql_type in ('DOUBLE', 'FLOAT', 'DECIMAL'):
        try:
            return str(float(val))
        except:
            return 'NULL'
    elif sql_type == 'BOOLEAN':
        return 'TRUE' if val else 'FALSE'
    else:
        return f"'{_escape_sql(val)}'"


def _get_or_create_table(df: pd.DataFrame, table_name: str) -> Dict[str, str]:
    """
    Get existing table schema or create new table from DataFrame.
    Returns column -> SQL type mapping.
    """
    # Check if table exists
    try:
        result = db_client.query(f"DESCRIBE TABLE {table_name}")
        if result:
            # Table exists - get column types
            col_types = {}
            for row in result:
                col_name = row.get('col_name', row.get('column_name', ''))
                data_type = row.get('data_type', row.get('type', 'STRING'))
                if col_name and not col_name.startswith('#'):  # Skip partition info
                    col_types[col_name.lower()] = data_type.upper()
            logger.info(f"Table exists with {len(col_types)} columns")
            return col_types
    except Exception as e:
        logger.info(f"Table doesn't exist, will create: {e}")
    
    # Table doesn't exist - create it
    columns = ['id STRING NOT NULL']
    col_types = {'id': 'STRING'}
    
    for col in df.columns:
        norm_col = _normalize_col(col)
        sql_type = _infer_sql_type(df[col])
        columns.append(f"{norm_col} {sql_type}")
        col_types[norm_col] = sql_type
    
    # Add metadata columns
    columns.append("upload_date TIMESTAMP")
    columns.append("source_file STRING")
    col_types['upload_date'] = 'TIMESTAMP'
    col_types['source_file'] = 'STRING'
    
    create_sql = f"CREATE TABLE IF NOT EXISTS {table_name} ({', '.join(columns)}) USING DELTA"
    logger.info(f"Creating table: {create_sql[:200]}...")
    db_client.execute(create_sql)
    logger.info(f"Table created with {len(col_types)} columns")
    
    return col_types


def process_churn_file(file_content: bytes, filename: str) -> Dict:
    """
    Process ANY uploaded data file (Excel or CSV).
    Fully dynamic - reads columns from file, creates/updates table, uploads data.
    """
    try:
        # Read file
        if filename.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(BytesIO(file_content))
        elif filename.endswith('.csv'):
            df = pd.read_csv(BytesIO(file_content))
        else:
            return {"success": False, "error": "Unsupported format. Use .xlsx, .xls, or .csv"}
        
        if df.empty:
            return {"success": False, "error": "File is empty"}
        
        logger.info(f"Loaded {len(df)} rows × {len(df.columns)} columns from {filename}")
        logger.info(f"Columns: {list(df.columns)}")
        
        # Normalize column names
        original_cols = list(df.columns)
        df.columns = [_normalize_col(c) for c in df.columns]
        col_mapping = dict(zip(df.columns, original_cols))
        
        # Get or create table
        col_types = _get_or_create_table(df, config.CHURN_TABLE)
        
        # Build batch values
        all_values = []
        file_cols = list(df.columns)
        
        for _, row in df.iterrows():
            try:
                values = [f"'{str(uuid.uuid4())}'"]  # id
                
                for col in file_cols:
                    sql_type = col_types.get(col, 'STRING')
                    val = row[col]
                    values.append(_format_value(val, sql_type))
                
                values.append("current_timestamp()")  # upload_date
                values.append(f"'{_escape_sql(filename)}'")  # source_file
                
                all_values.append(f"({', '.join(values)})")
            except Exception as e:
                logger.error(f"Row error: {e}")
        
        # Build column list for INSERT
        insert_cols = ['id'] + file_cols + ['upload_date', 'source_file']
        
        # Batch insert
        inserted = 0
        errors = 0
        batch_size = 500
        
        for i in range(0, len(all_values), batch_size):
            batch = all_values[i:i + batch_size]
            try:
                sql = f"INSERT INTO {config.CHURN_TABLE} ({', '.join(insert_cols)}) VALUES {', '.join(batch)}"
                db_client.execute(sql)
                inserted += len(batch)
                logger.info(f"Batch {i // batch_size + 1}: {len(batch)} rows (total: {inserted}/{len(all_values)})")
            except Exception as e:
                logger.error(f"Batch error: {e}")
                errors += len(batch)
        
        return {
            "success": True,
            "message": f"Uploaded {inserted} records from {filename}",
            "records_inserted": inserted,
            "errors": errors,
            "total_rows": len(df),
            "columns": file_cols,
            "column_types": {c: col_types.get(c, 'STRING') for c in file_cols}
        }
    
    except Exception as e:
        logger.error(f"Error processing file: {e}")
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}


def get_churn_data(limit: int = 1000) -> pd.DataFrame:
    """Get data from table as DataFrame"""
    try:
        sql = f"SELECT * FROM {config.CHURN_TABLE} ORDER BY upload_date DESC LIMIT {limit}"
        results = db_client.query(sql)
        return pd.DataFrame(results) if results else pd.DataFrame()
    except Exception as e:
        logger.error(f"Error getting data: {e}")
        return pd.DataFrame()


def get_table_schema() -> Dict:
    """Get the actual table schema dynamically"""
    try:
        result = db_client.query(f"DESCRIBE TABLE {config.CHURN_TABLE}")
        if result:
            columns = {}
            for row in result:
                col_name = row.get('col_name', row.get('column_name', ''))
                data_type = row.get('data_type', row.get('type', ''))
                if col_name and not col_name.startswith('#'):
                    columns[col_name] = data_type
            return {"success": True, "columns": columns}
        return {"success": True, "columns": {}}
    except Exception as e:
        return {"success": False, "error": str(e)}


def _get_churn_condition(churn_col: str, col_type: str) -> str:
    """
    Build SQL condition for churn detection based on column type.
    Handles both numeric (1/0) and string ('Yes'/'No', 'True'/'False') churn values.
    """
    if 'STRING' in col_type.upper():
        # String column - check for 'Yes', 'True', '1', etc.
        return f"(UPPER({churn_col}) IN ('YES', 'TRUE', '1', 'Y', 'CHURNED'))"
    else:
        # Numeric column
        return f"({churn_col} = 1)"


def get_churn_stats() -> Dict:
    """Get summary statistics - dynamically finds numeric columns and churn data"""
    try:
        # Get schema first
        schema = get_table_schema()
        if not schema.get('success'):
            return schema
        
        columns = schema.get('columns', {})
        
        if not columns:
            return {"success": True, "stats": {"total_customers": 0, "churn_rate": "0%"}}
        
        # Find numeric columns dynamically
        numeric_types = ('INT', 'BIGINT', 'DOUBLE', 'FLOAT', 'DECIMAL')
        numeric_cols = [c for c, t in columns.items() if any(nt in t.upper() for nt in numeric_types)]
        
        # Find churn column (look for common patterns)
        churn_col = None
        churn_col_type = 'STRING'
        for col, col_type in columns.items():
            col_lower = col.lower()
            if any(pattern in col_lower for pattern in ['churn_value', 'churnvalue', 'churn_label', 'churn', 'churned']):
                churn_col = col
                churn_col_type = col_type
                break
        
        # Build statistics query
        agg_parts = ["COUNT(*) as total_customers"]
        
        # Add churn-specific stats if churn column found
        if churn_col:
            churn_cond = _get_churn_condition(churn_col, churn_col_type)
            agg_parts.append(f"SUM(CASE WHEN {churn_cond} THEN 1 ELSE 0 END) as churned_customers")
            agg_parts.append(f"ROUND(100.0 * SUM(CASE WHEN {churn_cond} THEN 1 ELSE 0 END) / COUNT(*), 1) as churn_rate_pct")
        
        # Add numeric column stats
        for col in numeric_cols[:5]:  # Limit to 5 numeric columns for performance
            agg_parts.append(f"ROUND(AVG({col}), 2) as avg_{col}")
        
        sql = f"SELECT {', '.join(agg_parts)} FROM {config.CHURN_TABLE}"
        results = db_client.query(sql)
        
        if results:
            raw_stats = results[0]
            
            # Build formatted stats response
            stats = {
                "total_customers": raw_stats.get('total_customers', 0),
                "churn_rate": f"{raw_stats.get('churn_rate_pct', 0)}%" if raw_stats.get('churn_rate_pct') else "N/A",
                "churned_customers": raw_stats.get('churned_customers', 0) if churn_col else "N/A"
            }
            
            # Add average stats
            for key, value in raw_stats.items():
                if key.startswith('avg_'):
                    col_name = key[4:].replace('_', ' ').title()
                    if value is not None:
                        # Format as currency if it looks like a monetary value
                        if any(term in key.lower() for term in ['charge', 'cost', 'revenue', 'price']):
                            stats[f"avg_{key[4:]}"] = f"${value:,.2f}"
                        else:
                            stats[f"avg_{key[4:]}"] = f"{value:,.2f}"
            
            # Convert any datetime objects to strings for JSON serialization
            for key, value in stats.items():
                if hasattr(value, 'isoformat'):
                    stats[key] = str(value)
            
            return {
                "success": True, 
                "stats": stats, 
                "numeric_columns": numeric_cols,
                "churn_column": churn_col
            }
        return {"success": True, "stats": {"total_customers": 0, "churn_rate": "0%"}}
    
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        return {"success": False, "error": str(e)}


def get_churn_by_category(column: Optional[str] = None) -> Dict:
    """
    Get churn breakdown by category - dynamically finds string columns and calculates churn rates.
    If column specified, groups by that column.
    """
    try:
        schema = get_table_schema()
        if not schema.get('success'):
            return schema
        
        columns = schema.get('columns', {})
        
        # Find churn column and its type
        churn_col = None
        churn_col_type = 'STRING'
        for col, col_type in columns.items():
            col_lower = col.lower()
            if any(pattern in col_lower for pattern in ['churn_value', 'churnvalue', 'churn_label', 'churn', 'churned']):
                churn_col = col
                churn_col_type = col_type
                break
        
        # Find string columns (potential categories)
        string_cols = [c for c, t in columns.items() 
                       if 'STRING' in t.upper() and c.lower() not in ('id', 'source_file', 'customerid', 'customer_id')]
        
        results = {}
        
        # Priority columns for churn analysis
        priority_cols = ['contract', 'internet_service', 'payment_method', 'gender']
        
        # Find matching columns from schema
        cols_to_analyze = []
        for priority in priority_cols:
            for col in string_cols:
                if priority in col.lower():
                    cols_to_analyze.append(col)
                    break
        
        # Add any remaining string cols up to 5 total
        for col in string_cols:
            if col not in cols_to_analyze and len(cols_to_analyze) < 5:
                cols_to_analyze.append(col)
        
        # If specific column requested, use only that
        if column and column in string_cols:
            cols_to_analyze = [column]
        
        for col in cols_to_analyze:
            try:
                if churn_col:
                    # Calculate churn rate per category - handle both numeric and string churn values
                    churn_cond = _get_churn_condition(churn_col, churn_col_type)
                    sql = f"""
                        SELECT 
                            {col} as category, 
                            COUNT(*) as total,
                            SUM(CASE WHEN {churn_cond} THEN 1 ELSE 0 END) as churned,
                            ROUND(100.0 * SUM(CASE WHEN {churn_cond} THEN 1 ELSE 0 END) / COUNT(*), 1) as churn_rate
                        FROM {config.CHURN_TABLE}
                        WHERE {col} IS NOT NULL AND {col} != ''
                        GROUP BY {col}
                        ORDER BY total DESC
                        LIMIT 15
                    """
                else:
                    # No churn column - just count
                    sql = f"""
                        SELECT {col} as category, COUNT(*) as total
                        FROM {config.CHURN_TABLE}
                        WHERE {col} IS NOT NULL AND {col} != ''
                        GROUP BY {col}
                        ORDER BY total DESC
                        LIMIT 15
                    """
                
                query_results = db_client.query(sql) or []
                
                # Map results to expected format for charts
                # The chart expects: {category_col: value, 'total': count, 'churn_rate': rate}
                formatted_results = []
                for row in query_results:
                    formatted_row = {
                        col: row.get('category'),  # Add column-specific key
                        'category': row.get('category'),
                        'total': row.get('total', 0),
                        'count': row.get('total', 0),
                    }
                    if churn_col:
                        formatted_row['churned'] = row.get('churned', 0)
                        formatted_row['churn_rate'] = row.get('churn_rate', 0)
                    formatted_results.append(formatted_row)
                
                results[f"by_{col}"] = formatted_results
                
            except Exception as e:
                logger.error(f"Error analyzing {col}: {e}")
        
        return {
            "success": True, 
            "available_columns": string_cols,
            "churn_column": churn_col,
            **results
        }
    
    except Exception as e:
        logger.error(f"Error getting categories: {e}")
        return {"success": False, "error": str(e)}
