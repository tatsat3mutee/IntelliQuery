"""
HVS Handler - Upload and manage HVS (billing) data
"""

import uuid
import logging
import io
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd

from ..core.database import db_client
from ..core.config import config

logger = logging.getLogger(__name__)


def process_hvs_file(file_content: bytes, filename: str) -> Dict:
    """
    Process HVS CSV/Excel file and save to Databricks
    Expected columns: enterprise_id, subaccount, monthly_rate, etc.
    """
    try:
        # Read file based on extension
        if filename.endswith('.xlsx') or filename.endswith('.xls'):
            df = pd.read_excel(io.BytesIO(file_content))
        else:
            df = pd.read_csv(io.BytesIO(file_content))
        
        logger.info(f"Processing HVS file: {filename}, {len(df)} rows")
        logger.info(f"Columns found: {list(df.columns)}")
        
        # Standardize column names (lowercase, strip spaces)
        df.columns = [c.lower().strip().replace(' ', '_') for c in df.columns]
        
        # Insert each row
        count = 0
        errors = []
        
        for idx, row in df.iterrows():
            try:
                row_id = str(uuid.uuid4())
                
                # Extract values with defaults
                enterprise_id = str(row.get('enterprise_id', row.get('enterprise', '')))[:50]
                subaccount = str(row.get('subaccount', row.get('sub_account', '')))[:50]
                device_type = str(row.get('device_type', 'UNKNOWN'))[:50]
                service_type = str(row.get('service_type', 'UNKNOWN'))[:50]
                
                # Handle numeric values
                try:
                    monthly_rate = float(row.get('monthly_rate', row.get('rate', 0)))
                except:
                    monthly_rate = 0.0
                
                try:
                    bandwidth_mbps = int(row.get('bandwidth_mbps', row.get('bandwidth', 0)))
                except:
                    bandwidth_mbps = 0
                
                # Handle dates
                contract_start = row.get('contract_start', '2024-01-01')
                contract_end = row.get('contract_end', '2024-12-31')
                status = str(row.get('status', 'ACTIVE'))[:20]
                data_month = datetime.now().strftime('%Y-%m')
                
                # Escape strings
                enterprise_id = enterprise_id.replace("'", "''")
                subaccount = subaccount.replace("'", "''")
                device_type = device_type.replace("'", "''")
                service_type = service_type.replace("'", "''")
                
                sql = f"""
                    INSERT INTO {config.HVS_TABLE}
                    VALUES (
                        '{row_id}',
                        '{enterprise_id}',
                        '{subaccount}',
                        '{device_type}',
                        '{service_type}',
                        {monthly_rate},
                        {bandwidth_mbps},
                        '{contract_start}',
                        '{contract_end}',
                        '{status}',
                        current_timestamp(),
                        '{data_month}'
                    )
                """
                db_client.execute(sql)
                count += 1
                
            except Exception as e:
                errors.append(f"Row {idx}: {str(e)}")
                continue
        
        # Clean old data (keep only last N months)
        try:
            cleanup_sql = f"""
                DELETE FROM {config.HVS_TABLE}
                WHERE upload_date < date_sub(current_date(), {config.HVS_RETENTION_MONTHS * 30})
            """
            db_client.execute(cleanup_sql)
        except Exception as e:
            logger.warning(f"Cleanup warning: {e}")
        
        return {
            "success": True,
            "filename": filename,
            "records_saved": count,
            "errors": errors[:5] if errors else [],
            "message": f"HVS data uploaded! Saved {count} records."
        }
    
    except Exception as e:
        logger.error(f"Error processing HVS file: {e}")
        return {
            "success": False,
            "error": str(e)
        }


def get_hvs_summary() -> pd.DataFrame:
    """Get HVS data summary for charts"""
    try:
        sql = f"""
            SELECT 
                enterprise_id,
                subaccount,
                COUNT(*) as record_count,
                AVG(monthly_rate) as avg_rate,
                SUM(monthly_rate) as total_rate,
                AVG(bandwidth_mbps) as avg_bandwidth,
                MIN(upload_date) as first_upload,
                MAX(upload_date) as last_upload
            FROM {config.HVS_TABLE}
            GROUP BY enterprise_id, subaccount
            ORDER BY total_rate DESC
            LIMIT 50
        """
        results = db_client.query(sql)
        
        if results:
            return pd.DataFrame(results)
        return pd.DataFrame()
    
    except Exception as e:
        logger.error(f"HVS summary error: {e}")
        return pd.DataFrame()


def get_hvs_stats() -> Dict:
    """Get HVS data statistics"""
    try:
        sql = f"""
            SELECT 
                COUNT(*) as total_records,
                COUNT(DISTINCT enterprise_id) as unique_enterprises,
                COUNT(DISTINCT subaccount) as unique_subaccounts,
                AVG(monthly_rate) as avg_rate,
                MIN(monthly_rate) as min_rate,
                MAX(monthly_rate) as max_rate,
                SUM(monthly_rate) as total_revenue,
                MIN(upload_date) as first_upload,
                MAX(upload_date) as last_upload
            FROM {config.HVS_TABLE}
        """
        results = db_client.query(sql)
        
        if results:
            return {"success": True, "stats": results[0]}
        return {"success": True, "stats": {}}
    
    except Exception as e:
        logger.error(f"HVS stats error: {e}")
        return {"success": False, "error": str(e)}


def get_hvs_raw_data(limit: int = 1000) -> pd.DataFrame:
    """Get raw HVS data for ML training"""
    try:
        sql = f"""
            SELECT 
                enterprise_id,
                subaccount,
                device_type,
                service_type,
                monthly_rate,
                bandwidth_mbps,
                status,
                data_month
            FROM {config.HVS_TABLE}
            WHERE monthly_rate > 0
            ORDER BY upload_date DESC
            LIMIT {limit}
        """
        results = db_client.query(sql)
        
        if results:
            return pd.DataFrame(results)
        return pd.DataFrame()
    
    except Exception as e:
        logger.error(f"HVS raw data error: {e}")
        return pd.DataFrame()
