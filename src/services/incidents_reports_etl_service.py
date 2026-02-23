import duckdb
import os
import pandas as pd

REPORTS_DATA_PATH = os.getenv("REPORTS_DATA_PATH", "data/raw/reports.csv")
DB_PATH = os.getenv("DUCKDB_PATH", "data/duckdb/reports.duckdb")

def create_reports_table():
    if DB_PATH != ':memory:':
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    
    con = duckdb.connect(database=DB_PATH, read_only=False)
    
    con.execute("""
        CREATE TABLE IF NOT EXISTS reports (
            _id VARCHAR,
            authors VARCHAR,
            date_downloaded VARCHAR,
            date_modified VARCHAR,
            date_published VARCHAR,
            date_submitted VARCHAR,
            description VARCHAR,
            epoch_date_downloaded BIGINT,
            epoch_date_modified BIGINT,
            epoch_date_published BIGINT,
            epoch_date_submitted BIGINT,
            image_url VARCHAR,
            language VARCHAR,
            ref_number BIGINT,
            report_number BIGINT,
            source_domain VARCHAR,
            submitters VARCHAR,
            text VARCHAR,
            title VARCHAR,
            url VARCHAR,
            tags VARCHAR
        )
    """)
    
    count = con.execute("SELECT COUNT(*) FROM reports").fetchone()[0] #type: ignore
    
    if count == 0:
        if os.path.exists(REPORTS_DATA_PATH):
            print(f"Loading data from {REPORTS_DATA_PATH}...")
            try:
                df = pd.read_csv(REPORTS_DATA_PATH)

                numeric_cols = [
                    'epoch_date_downloaded', 'epoch_date_modified', 
                    'epoch_date_published', 'epoch_date_submitted',
                    'ref_number', 'report_number'
                ]
                
                for col in numeric_cols:
                    if col in df.columns:
                        # Replace specific JSON NaN representation and other non-numeric values
                        df[col] = df[col].astype(str).replace(r"\{'\$numberDouble': 'NaN'\}", None, regex=True)
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                
                con.register('df_reports', df)
                con.execute("INSERT INTO reports SELECT * FROM df_reports")
                print("Data loaded successfully.")
            except Exception as e:
                print(f"Error loading data: {e}")
        else:
            print(f"Warning: Data file not found at {REPORTS_DATA_PATH}")
    
    return con

def get_reports_by_ids(row_ids: list[int]):
    if not row_ids:
        return []
    
    con = duckdb.connect(database=DB_PATH, read_only=True)
    try:
        ids_str = ','.join(map(str, row_ids))
        
        query = f"""
            SELECT 
                rowid,
                authors as Author, 
                date_published, 
                description, 
                image_url, 
                language, 
                source_domain, 
                title, 
                text, 
                url 
            FROM reports 
            WHERE rowid IN ({ids_str})
        """
        
        df = con.execute(query).fetchdf()
        
        if df.empty:
            return []

        id_map = {row_id: i for i, row_id in enumerate(row_ids)}
        df['sort_order'] = df['rowid'].map(id_map)
        df = df.sort_values('sort_order')
        
        result_df = df.drop(columns=['rowid', 'sort_order'])
        
        return result_df.to_dict(orient='records')
        
    except Exception as e:
        print(f"Error retrieving reports: {e}")
        return []
    finally:
        con.close()


