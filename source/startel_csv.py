from pyspark.sql import SparkSession
from pyspark.sql.functions import regexp_extract, udf, col
from pyspark.sql.types import StringType, DoubleType
import pyspark.sql.functions as F
import fitz 

spark = SparkSession.builder.appName("StartelInvoiceETL").getOrCreate()

# ========= INPUT / OUTPUT =========
input_s3_path = "s3://startel/startel_invoices/"
output_s3_path = "s3://startel/startel_output_csv/"

# ========= READ PDF FILES =========
pdf_df = (
    spark.read.format("binaryFile")
    .option("pathGlobFilter", "*.pdf")
    .load(input_s3_path)
)

# ========= PDF TEXT EXTRACTION =========
def extract_text(content):
    try:
        doc = fitz.open(stream=content, filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return text
    except Exception:
        return ""

extract_text_udf = udf(extract_text, StringType())

text_df = pdf_df.withColumn("text", extract_text_udf(col("content")))

# ========= REGEX EXTRACTION =========
final_df = text_df.select(
    regexp_extract("text", r"(CUST\d+)", 1).alias("customer_id"),

    regexp_extract(
        "text",
        r"Customer\s*Name\s*[:\-]?\s*([A-Za-z ]+)",
        1
    ).alias("customer_name"),

    regexp_extract(
        "text",
        r"City\s*[:\-]?\s*([A-Za-z ]+)",
        1
    ).alias("city"),

    regexp_extract(
        "text",
        r"Plan\s*[:\-]?\s*([A-Za-z]+)",
        1
    ).alias("plan"),

    regexp_extract(
        "text",
        r"Billing\s*Month\s*[:\-]?\s*([A-Za-z]+\s+\d{4})",
        1
    ).alias("billing_month"),

    # -------- USAGE CHARGES (robust) --------
    regexp_extract(
        "text",
        r"(Usage\s*Charges|Total\s*Usage)[\s\S]{0,50}?([0-9,]+\.\d{2})",
        2
    ).alias("usage_charges"),

    # -------- GST (handles GST / CGST+SGST) --------
    regexp_extract(
        "text",
        r"(GST|CGST|SGST)[\s\S]{0,50}?([0-9,]+\.\d{2})",
        2
    ).alias("gst_amount"),

    # -------- BILL DUE / TOTAL --------
    regexp_extract(
        "text",
        r"(Total\s*Amount\s*Due|Bill\s*Due|Net\s*Payable)[\s\S]{0,50}?([0-9,]+\.\d{2})",
        2
    ).alias("bill_due")
)


# ========= TYPE CAST =========
final_df = (
    final_df
    .withColumn("usage_charges", F.regexp_replace("usage_charges", ",", "").cast(DoubleType()))
    .withColumn("gst_amount", F.regexp_replace("gst_amount", ",", "").cast(DoubleType()))
    .withColumn("bill_due", F.col("usage_charges") + F.col("gst_amount"))
)

# ===== WRITE CSV =========
(
    final_df
    .coalesce(1)
    .write
    .mode("overwrite")
    .option("header", True)
    .csv(output_s3_path)
)

spark.stop()
