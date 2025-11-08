import pandas as pd # type: ignore
import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore
import seaborn as sns # type: ignore
from matplotlib.gridspec import GridSpec # type: ignore

features = [
	'having_IP_Address', 'URL_Length', 'Shortining_Service', 'having_At_Symbol', 'double_slash_redirecting',
	'Prefix_Suffix', 'having_Sub_Domain', 'SSLfinal_State', 'Domain_registeration_length', 'Favicon', 'port',
	'HTTPS_token', 'Request_URL', 'URL_of_Anchor', 'Links_in_tags', 'SFH', 'Submitting_to_email', 'Abnormal_URL',
	'Redirect', 'on_mouseover', 'RightClick', 'popUpWidnow', 'Iframe', 'age_of_domain', 'DNSRecord', 'web_traffic',
	'Page_Rank', 'Google_Index', 'Links_pointing_to_page', 'Statistical_report'
]

df = pd.read_csv('Main/training_dataset.csv')

# Set style for better visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


# ============================================
# 0. DISPLAY DATASET HEAD
# ============================================
def display_dataset_head(dataframe, n_rows=10, method='html'):
	"""
	Display the first n rows of the dataset in a readable format
	
	Parameters:
	- dataframe: pandas DataFrame
	- n_rows: number of rows to display (default 10)
	- method: 'html' (opens in browser), 'excel' (saves as Excel), or 'styled' (matplotlib table)
	"""
	print("\n" + "="*60)
	print(f"DISPLAYING FIRST {n_rows} ROWS OF DATASET")
	print("="*60)
	
	head_data = dataframe.head(n_rows)
	
	if method == 'html':
		# Create a styled HTML table and open in browser
		import webbrowser
		import os
		
		html_content = f"""
		<!DOCTYPE html>
		<html>
		<head>
			<title>Dataset Head - First {n_rows} Rows</title>
			<style>
				body {{
					font-family: Arial, sans-serif;
					margin: 20px;
					background-color: #f5f5f5;
				}}
				h2 {{
					color: #2c3e50;
					text-align: center;
				}}
				.info {{
					background-color: #3498db;
					color: white;
					padding: 10px;
					border-radius: 5px;
					margin-bottom: 20px;
					text-align: center;
				}}
				table {{
					border-collapse: collapse;
					width: 100%;
					background-color: white;
					box-shadow: 0 2px 5px rgba(0,0,0,0.1);
					font-size: 12px;
				}}
				th {{
					background-color: #34495e;
					color: white;
					padding: 12px;
					text-align: center;
					position: sticky;
					top: 0;
					z-index: 10;
				}}
				td {{
					border: 1px solid #ddd;
					padding: 10px;
					text-align: center;
				}}
				tr:nth-child(even) {{
					background-color: #f2f2f2;
				}}
				tr:hover {{
					background-color: #e8f4f8;
				}}
				.index-col {{
					background-color: #ecf0f1;
					font-weight: bold;
				}}
				.positive {{
					color: #27ae60;
					font-weight: bold;
				}}
				.negative {{
					color: #e74c3c;
					font-weight: bold;
				}}
			</style>
		</head>
		<body>
			<h2>📊 Dataset Preview - First {n_rows} Rows</h2>
			<div class="info">
				<strong>Total Rows:</strong> {len(dataframe)} | 
				<strong>Total Columns:</strong> {len(dataframe.columns)} | 
				<strong>Shape:</strong> {dataframe.shape}
			</div>
			<div style="overflow-x: auto;">
		"""
		
		# Convert DataFrame to HTML with custom styling
		html_table = head_data.to_html(index=True, classes='data-table', border=0)
		
		# Add color coding for Result column
		html_table = html_table.replace('<td>1</td>', '<td class="positive">1</td>')
		html_table = html_table.replace('<td>-1</td>', '<td class="negative">-1</td>')
		
		html_content += html_table + """
			</div>
		</body>
		</html>
		"""
		
		# Save to temporary HTML file
		html_file = 'dataset_head.html'
		with open(html_file, 'w', encoding='utf-8') as f:
			f.write(html_content)
		
		# Open in browser
		webbrowser.open('file://' + os.path.abspath(html_file))
		print(f"✓ Dataset head opened in browser: {html_file}")
		
	elif method == 'excel':
		# Save to Excel file
		excel_file = 'dataset_head.xlsx'
		with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
			head_data.to_excel(writer, sheet_name='Dataset Head', index=True)
			
			# Auto-adjust column widths
			worksheet = writer.sheets['Dataset Head']
			for idx, col in enumerate(head_data.columns, start=2):
				max_length = max(
					head_data[col].astype(str).apply(len).max(),
					len(str(col))
				)
				worksheet.column_dimensions[chr(64 + idx)].width = min(max_length + 2, 20)
		
		print(f"✓ Dataset head saved to Excel: {excel_file}")
		print("  You can open this file in Excel or any spreadsheet application")
		
	elif method == 'styled':
		# Create a matplotlib table visualization
		fig, ax = plt.subplots(figsize=(16, max(6, n_rows * 0.5)))
		ax.axis('tight')
		ax.axis('off')
		
		# Prepare data for table
		table_data = []
		table_data.append(list(head_data.columns))
		for idx, row in head_data.iterrows():
			table_data.append([str(idx)] + [str(val) for val in row.values])
		
		# Create table
		table = ax.table(cellText=table_data[1:], colLabels=['Index'] + list(head_data.columns),
						cellLoc='center', loc='center')
		
		table.auto_set_font_size(False)
		table.set_fontsize(8)
		table.scale(1, 2)
		
		# Style header
		for i in range(len(head_data.columns) + 1):
			table[(0, i)].set_facecolor('#34495e')
			table[(0, i)].set_text_props(weight='bold', color='white')
		
		# Style rows
		for i in range(1, len(table_data)):
			for j in range(len(head_data.columns) + 1):
				if i % 2 == 0:
					table[(i, j)].set_facecolor('#f2f2f2')
		
		plt.title(f'Dataset Head - First {n_rows} Rows', fontsize=14, fontweight='bold', pad=20)
		plt.tight_layout()
		plt.savefig('dataset_head_table.png', dpi=300, bbox_inches='tight')
		plt.show()
		print(f"✓ Dataset head table saved as image: dataset_head_table.png")
	
	# Also print basic info
	print(f"\nDataset Shape: {dataframe.shape}")
	print(f"Total Features: {len(dataframe.columns) - 1}")  # Excluding target
	print(f"Total Samples: {len(dataframe)}")
	print(f"\nColumn Names:")
	for i, col in enumerate(dataframe.columns, 1):
		print(f"  {i}. {col}")


# ============================================
# 1. MISSING VALUES / NULL CHECK
# ============================================
def check_missing_values(dataframe):
	"""Check for missing values and null values in the dataset"""
	print("\n" + "="*60)
	print("MISSING VALUES ANALYSIS")
	print("="*60)
	
	# Count missing values
	missing_count = dataframe.isnull().sum()
	missing_percentage = (dataframe.isnull().sum() / len(dataframe)) * 100
	
	missing_df = pd.DataFrame({
		'Column': missing_count.index,
		'Missing_Count': missing_count.values,
		'Missing_Percentage': missing_percentage.values
	})
	
	missing_df = missing_df[missing_df['Missing_Count'] > 0].sort_values('Missing_Count', ascending=False)
	
	if len(missing_df) == 0:
		print("✓ No missing values found in the dataset!")
	else:
		print(f"✗ Found missing values in {len(missing_df)} columns:")
		print(missing_df.to_string(index=False))
	
	print(f"\nTotal Records: {len(dataframe)}")
	print(f"Total Missing Values: {dataframe.isnull().sum().sum()}")
	
	return missing_df


# ============================================
# 2. DATASET BALANCE CHECK
# ============================================
def check_balance(dataframe, target_column='Result'):
	"""Check if the dataset is balanced"""
	print("\n" + "="*60)
	print("DATASET BALANCE ANALYSIS")
	print("="*60)
	
	# Count each class
	value_counts = dataframe[target_column].value_counts()
	percentages = (value_counts / len(dataframe)) * 100
	
	print(f"\nClass Distribution:")
	for value, count in value_counts.items():
		percentage = percentages[value]
		print(f"  Class {value}: {count} samples ({percentage:.2f}%)")
	
	# Calculate balance ratio
	min_class = value_counts.min()
	max_class = value_counts.max()
	balance_ratio = min_class / max_class
	
	print(f"\nBalance Ratio: {balance_ratio:.4f}")
	
	if balance_ratio >= 0.8:
		print("✓ Dataset is BALANCED (ratio >= 0.8)")
	elif balance_ratio >= 0.5:
		print("⚠ Dataset is MODERATELY IMBALANCED (0.5 <= ratio < 0.8)")
	else:
		print("✗ Dataset is HIGHLY IMBALANCED (ratio < 0.5)")
	
	return value_counts, balance_ratio


# ============================================
# 3. CORRELATION ANALYSIS
# ============================================
def analyze_correlation(dataframe, features_list, target_column='Result'):
	"""Analyze correlation between features and target"""
	print("\n" + "="*60)
	print("CORRELATION ANALYSIS")
	print("="*60)
	
	# Calculate correlation matrix
	correlation_matrix = dataframe[features_list + [target_column]].corr()
	
	# Get correlation with target
	target_correlation = correlation_matrix[target_column].drop(target_column).sort_values(ascending=False)
	
	print(f"\nALL Features Correlation with {target_column} (Sorted by Correlation):")
	print("="*60)
	for idx, (feature, corr_value) in enumerate(target_correlation.items(), 1):
		print(f"{idx:2d}. {feature:35s}: {corr_value:7.4f}")
	
	print(f"\nTop 10 Features Most Correlated with {target_column}:")
	print(target_correlation.head(10))
	
	print(f"\nBottom 10 Features Least Correlated with {target_column}:")
	print(target_correlation.tail(10))
	
	# Find highly correlated feature pairs
	print("\nHighly Correlated Feature Pairs (|correlation| > 0.7):")
	high_corr_pairs = []
	for i in range(len(features_list)):
		for j in range(i+1, len(features_list)):
			corr_value = correlation_matrix.loc[features_list[i], features_list[j]]
			if abs(corr_value) > 0.7:
				high_corr_pairs.append((features_list[i], features_list[j], corr_value))
	
	if high_corr_pairs:
		for feat1, feat2, corr in sorted(high_corr_pairs, key=lambda x: abs(x[2]), reverse=True):
			print(f"  {feat1} <-> {feat2}: {corr:.4f}")
	else:
		print("  No highly correlated feature pairs found.")
	
	return correlation_matrix, target_correlation


# ============================================
# 4. GRAPH VISUALIZATIONS
# ============================================
def create_visualizations(dataframe, features_list, target_column='Result', save_plots=True):
	"""Create comprehensive visualizations for the dataset"""
	print("\n" + "="*60)
	print("GENERATING VISUALIZATIONS")
	print("="*60)
	
	# 1. Class Distribution Bar Plot
	plt.figure(figsize=(10, 6))
	value_counts = dataframe[target_column].value_counts()
	colors = ['#2ecc71' if val == 1 else '#e74c3c' for val in value_counts.index]
	plt.bar(value_counts.index, value_counts.values, color=colors, edgecolor='black', linewidth=1.5)
	plt.xlabel('Class', fontsize=12, fontweight='bold')
	plt.ylabel('Count', fontsize=12, fontweight='bold')
	plt.title('Class Distribution (Target Variable)', fontsize=14, fontweight='bold')
	plt.xticks(value_counts.index)
	for i, v in enumerate(value_counts.values):
		plt.text(value_counts.index[i], v + 50, str(v), ha='center', fontweight='bold')
	plt.tight_layout()
	if save_plots:
		plt.savefig('class_distribution.png', dpi=300, bbox_inches='tight')
	plt.show()
	print("✓ Generated: Class Distribution Plot")
	
	# 2. Correlation Heatmap (ALL Features)
	correlation_matrix = dataframe[features_list + [target_column]].corr()
	target_corr = correlation_matrix[target_column].drop(target_column).abs().sort_values(ascending=False)
	
	# Create heatmap with ALL features
	plt.figure(figsize=(20, 18))
	sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
				center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.8},
				annot_kws={'size': 6})
	plt.title('Correlation Heatmap (ALL Features)', fontsize=16, fontweight='bold')
	plt.xticks(rotation=90, ha='right', fontsize=8)
	plt.yticks(rotation=0, fontsize=8)
	plt.tight_layout()
	if save_plots:
		plt.savefig('correlation_heatmap_all.png', dpi=300, bbox_inches='tight')
	plt.show()
	print("✓ Generated: Correlation Heatmap (All Features)")
	
	# Also create a focused heatmap for top 15 features for better readability
	top_features = target_corr.head(15).index.tolist() + [target_column]
	plt.figure(figsize=(14, 12))
	sns.heatmap(dataframe[top_features].corr(), annot=True, fmt='.2f', cmap='coolwarm', 
				center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
	plt.title('Correlation Heatmap (Top 15 Most Correlated Features)', fontsize=14, fontweight='bold')
	plt.tight_layout()
	if save_plots:
		plt.savefig('correlation_heatmap_top15.png', dpi=300, bbox_inches='tight')
	plt.show()
	print("✓ Generated: Correlation Heatmap (Top 15 Features)")
	
	# 3. Feature Correlation with Target
	plt.figure(figsize=(12, 8))
	target_correlation = correlation_matrix[target_column].drop(target_column).sort_values()
	colors = ['#e74c3c' if x < 0 else '#2ecc71' for x in target_correlation.values]
	target_correlation.plot(kind='barh', color=colors, edgecolor='black')
	plt.xlabel('Correlation Coefficient', fontsize=12, fontweight='bold')
	plt.ylabel('Features', fontsize=12, fontweight='bold')
	plt.title('Feature Correlation with Target Variable', fontsize=14, fontweight='bold')
	plt.axvline(x=0, color='black', linestyle='--', linewidth=1)
	plt.tight_layout()
	if save_plots:
		plt.savefig('feature_correlation.png', dpi=300, bbox_inches='tight')
	plt.show()
	print("✓ Generated: Feature Correlation Bar Chart")
	
	# 4. Feature Distribution (ALL features in multiple pages)
	print("\n✓ Generating Feature Distribution Plots for ALL features...")
	
	# Create distribution plots for all features (5 features per page)
	features_per_page = 6
	n_pages = (len(features_list) + features_per_page - 1) // features_per_page
	
	for page in range(n_pages):
		start_idx = page * features_per_page
		end_idx = min((page + 1) * features_per_page, len(features_list))
		page_features = features_list[start_idx:end_idx]
		
		n_features = len(page_features)
		n_cols = 3
		n_rows = (n_features + n_cols - 1) // n_cols
		
		fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 4))
		if n_rows == 1:
			axes = axes.reshape(1, -1)
		axes = axes.ravel()
		
		for idx, feature in enumerate(page_features):
			dataframe[feature].value_counts().sort_index().plot(kind='bar', ax=axes[idx], 
																color='steelblue', edgecolor='black')
			axes[idx].set_title(f'{feature}', fontweight='bold', fontsize=10)
			axes[idx].set_xlabel('Value', fontsize=9)
			axes[idx].set_ylabel('Count', fontsize=9)
		
		# Hide extra subplots
		for idx in range(len(page_features), len(axes)):
			axes[idx].axis('off')
		
		plt.suptitle(f'Feature Distributions (Page {page + 1}/{n_pages})', fontsize=16, fontweight='bold')
		plt.tight_layout()
		if save_plots:
			plt.savefig(f'feature_distributions_page{page + 1}.png', dpi=300, bbox_inches='tight')
		plt.show()
	
	print(f"✓ Generated: {n_pages} pages of Feature Distribution Plots")
	
	# 5. Box Plot for Feature Values by Target Class (ALL features)
	print("\n✓ Generating Box Plots for ALL features...")
	
	# Create box plots for all features (6 features per page)
	features_per_page = 6
	n_pages = (len(features_list) + features_per_page - 1) // features_per_page
	
	for page in range(n_pages):
		start_idx = page * features_per_page
		end_idx = min((page + 1) * features_per_page, len(features_list))
		page_features = features_list[start_idx:end_idx]
		
		n_features = len(page_features)
		n_cols = 3
		n_rows = (n_features + n_cols - 1) // n_cols
		
		fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 4))
		if n_rows == 1:
			axes = axes.reshape(1, -1)
		axes = axes.ravel()
		
		for idx, feature in enumerate(page_features):
			dataframe.boxplot(column=feature, by=target_column, ax=axes[idx])
			axes[idx].set_title(f'{feature}', fontweight='bold', fontsize=10)
			axes[idx].set_xlabel('Target Class', fontsize=9)
			axes[idx].set_ylabel('Value', fontsize=9)
			plt.sca(axes[idx])
			plt.xticks([1, 2], ['-1', '1'])
		
		# Hide extra subplots
		for idx in range(len(page_features), len(axes)):
			axes[idx].axis('off')
		
		plt.suptitle(f'Feature Value Distribution by Target Class (Page {page + 1}/{n_pages})', 
					fontsize=16, fontweight='bold')
		plt.tight_layout()
		if save_plots:
			plt.savefig(f'feature_boxplots_page{page + 1}.png', dpi=300, bbox_inches='tight')
		plt.show()
	
	print(f"✓ Generated: {n_pages} pages of Box Plots by Target Class")
	
	print("\nAll visualizations saved successfully!")


# ============================================
# MAIN EXECUTION
# ============================================
print("="*60)
print("DATASET ANALYSIS REPORT")
print("="*60)

print("\nFeatures  Description: ") 
for feature in features:
	describe_output = df[feature].describe()
	print(describe_output)
	print()
      
describe_target = df['Result'].describe()
print("Target Description: ")
print(describe_target)

# Execute all analysis functions
display_dataset_head(df, n_rows=10, method='html')  # Opens in browser for better viewing
missing_analysis = check_missing_values(df)
balance_info = check_balance(df, target_column='Result')
correlation_matrix, target_corr = analyze_correlation(df, features, target_column='Result')
create_visualizations(df, features, target_column='Result', save_plots=True)

print("\n" + "="*60)
print("ANALYSIS COMPLETE!")
print("="*60)


