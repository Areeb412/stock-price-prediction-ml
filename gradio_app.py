"""
gradio_app.py
-------------
Gradio interface for the stock price predictor.

Run with:
    uv run python gradio_app.py

Then open: http://localhost:7860
"""

import logging
import warnings
from datetime import datetime, timedelta

import gradio as gr
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Shared ML backend modules
from src.data_loader import fetch_and_validate
from src.features import engineer_features, get_feature_columns
from src.model_trainer import train_and_evaluate
from src.predictor import predict_next_day

# Setup
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# Default values shown in the UI when the app first loads
DEFAULT_TICKER = "AAPL"
DEFAULT_START_DATE = (datetime.today() - timedelta(days=5 * 365)).strftime("%Y-%m-%d")
DEFAULT_END_DATE = datetime.today().strftime("%Y-%m-%d")
DEFAULT_MODEL = "Random Forest"
DEFAULT_TRAIN_RATIO = 0.80

# Prediction pipeline
def run_prediction_pipeline(
    ticker: str,
    start_date: str,
    end_date: str,
    model_choice: str,
    train_ratio: float,
    n_trees: int,
    max_depth: int,
) -> tuple:
    """
    Run the data, feature, training, and prediction steps for the Gradio UI.

    Inputs and outputs line up with the Gradio widgets.

    Returns (in order):
        1. Status message string
        2. Plotly figure: Actual vs Predicted
        3. Plotly figure: Feature Importances
        4. Pandas DataFrame: Evaluation metrics
        5. Pandas DataFrame: next-day prediction
        6. Plotly figure: EDA (price history + volume)
    """
    ticker = ticker.strip().upper()

    try:
        # Fetch data
        status_msg = f"Fetching {ticker} data from Yahoo Finance..."
        yield status_msg, None, None, None, None, None

        df_raw = fetch_and_validate(ticker, start_date, end_date)
        df_raw.attrs["ticker"] = ticker

        # Engineer features
        status_msg = f"Engineering features for {ticker}..."
        yield status_msg, None, None, None, None, None

        df_feat = engineer_features(df_raw)
        feat_cols = get_feature_columns(df_feat)

        # Train models
        status_msg = f"Training {model_choice} model on {int(train_ratio*100)}% of data..."
        yield status_msg, None, None, None, None, None

        results = train_and_evaluate(
            df_features=df_feat,
            feature_cols=feat_cols,
            train_ratio=train_ratio,
            rf_n_estimators=n_trees,
            rf_max_depth=max_depth,
        )

        # Generate next-day prediction
        prediction = predict_next_day(results, df_feat)

        # Build plots
        status_msg = f"Rendering charts for {ticker}..."
        yield status_msg, None, None, None, None, None

        # Map UI choice to results key
        model_key_map = {
            "Linear Regression": ("linear_regression", "LR_Predicted"),
            "Random Forest":     ("random_forest",     "RF_Predicted"),
            "Ensemble":          ("ensemble",           "Ens_Predicted"),
        }
        metrics_key, pred_col = model_key_map[model_choice]

        test_df = results["test_df"]
        # Price history and volume
        eda_fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            row_heights=[0.70, 0.30],
            subplot_titles=[f"{ticker} Historical Closing Price", "Daily Volume"],
            vertical_spacing=0.07,
        )

        # Moving averages
        ma50 = df_raw["Close"].rolling(50).mean()
        ma200 = df_raw["Close"].rolling(200).mean()

        eda_fig.add_trace(
            go.Scatter(
                x=df_raw.index, y=df_raw["Close"],
                name="Close Price",
                line=dict(color="#2196F3", width=1.2),
                fill="tozeroy", fillcolor="rgba(33,150,243,0.05)",
            ), row=1, col=1,
        )
        eda_fig.add_trace(
            go.Scatter(x=df_raw.index, y=ma50, name="50-Day MA",
                       line=dict(color="#FF9800", width=1.4, dash="dash")),
            row=1, col=1,
        )
        eda_fig.add_trace(
            go.Scatter(x=df_raw.index, y=ma200, name="200-Day MA",
                       line=dict(color="#F44336", width=1.4, dash="dot")),
            row=1, col=1,
        )
        eda_fig.add_trace(
            go.Bar(x=df_raw.index, y=df_raw["Volume"] / 1e6,
                   name="Volume (M)", marker_color="#4CAF50", opacity=0.5),
            row=2, col=1,
        )
        eda_fig.update_layout(
            height=550, template="plotly_dark",
            title=dict(text=f"{ticker} Historical Analysis", font_size=18),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            hovermode="x unified",
        )

        # Actual vs predicted
        pred_fig = go.Figure()

        pred_fig.add_trace(go.Scatter(
            x=test_df.index, y=test_df["Target_Close"],
            name="Actual Close",
            line=dict(color="#2196F3", width=2),
        ))
        pred_fig.add_trace(go.Scatter(
            x=test_df.index, y=test_df[pred_col],
            name=f"{model_choice} Prediction",
            line=dict(color="#4CAF50", width=1.8, dash="dash"),
        ))
        # Error band: shade region between actual and predicted
        pred_fig.add_trace(go.Scatter(
            x=pd.concat([test_df.index.to_series(), test_df.index.to_series()[::-1]]),
            y=pd.concat([
                test_df["Target_Close"],
                test_df[pred_col][::-1],
            ]),
            fill="toself",
            fillcolor="rgba(255,152,0,0.12)",
            line=dict(color="rgba(255,255,255,0)"),
            name="Error Band",
        ))
        pred_fig.update_layout(
            height=420, template="plotly_dark",
            title=dict(text=f"{ticker} Actual vs {model_choice} Predicted", font_size=16),
            yaxis_title="Price (USD)",
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )

        # Feature importances
        fi_data = results["feature_importance"]
        fi_df = pd.DataFrame(fi_data).sort_values("importance")
        max_imp = fi_df["importance"].max()
        bar_colors = [
            f"rgba(76,175,80,{0.4 + 0.6 * (v / max_imp)})"
            for v in fi_df["importance"]
        ]

        fi_fig = go.Figure(go.Bar(
            x=fi_df["importance"],
            y=fi_df["feature"],
            orientation="h",
            marker_color=bar_colors,
            text=[f"{v:.4f}" for v in fi_df["importance"]],
            textposition="outside",
        ))
        fi_fig.update_layout(
            height=500, template="plotly_dark",
            title=dict(text="Random Forest Feature Importances (Top 15)", font_size=16),
            xaxis_title="Importance Score",
            margin=dict(l=20, r=80, t=60, b=20),
        )

        # Metrics table
        all_metrics = results["metrics"]
        metrics_display = pd.DataFrame([
            {
                "Model": "Linear Regression",
                "MAE ($)": all_metrics["linear_regression"]["mae"],
                "RMSE ($)": all_metrics["linear_regression"]["rmse"],
                "R2": all_metrics["linear_regression"]["r2"],
                "MAPE (%)": all_metrics["linear_regression"]["mape"],
                "Dir Acc (%)": all_metrics["linear_regression"]["directional_accuracy"],
            },
            {
                "Model": "Random Forest",
                "MAE ($)": all_metrics["random_forest"]["mae"],
                "RMSE ($)": all_metrics["random_forest"]["rmse"],
                "R2": all_metrics["random_forest"]["r2"],
                "MAPE (%)": all_metrics["random_forest"]["mape"],
                "Dir Acc (%)": all_metrics["random_forest"]["directional_accuracy"],
            },
            {
                "Model": "Ensemble",
                "MAE ($)": all_metrics["ensemble"]["mae"],
                "RMSE ($)": all_metrics["ensemble"]["rmse"],
                "R2": all_metrics["ensemble"]["r2"],
                "MAPE (%)": all_metrics["ensemble"]["mape"],
                "Dir Acc (%)": all_metrics["ensemble"]["directional_accuracy"],
            },
        ])

        # Next-day prediction table
        preds = prediction["predictions"]
        pred_display = pd.DataFrame([
            {
                "Model":              "Linear Regression",
                "Current Close ($)":   prediction["current_close"],
                "Next Close ($)": preds["linear_regression"]["predicted_close"],
                "Change ($)":          preds["linear_regression"]["change_dollars"],
                "Change (%)":          preds["linear_regression"]["change_percent"],
                "Direction":           preds["linear_regression"]["direction"],
            },
            {
                "Model":              "Random Forest",
                "Current Close ($)":   prediction["current_close"],
                "Next Close ($)": preds["random_forest"]["predicted_close"],
                "Change ($)":          preds["random_forest"]["change_dollars"],
                "Change (%)":          preds["random_forest"]["change_percent"],
                "Direction":           preds["random_forest"]["direction"],
            },
            {
                "Model":              "Ensemble",
                "Current Close ($)":   prediction["current_close"],
                "Next Close ($)": preds["ensemble"]["predicted_close"],
                "Change ($)":          preds["ensemble"]["change_dollars"],
                "Change (%)":          preds["ensemble"]["change_percent"],
                "Direction":           preds["ensemble"]["direction"],
            },
        ])

        sel_m = all_metrics[metrics_key]
        final_status = (
            f"Done. {ticker} | {model_choice} | "
            f"MAE: ${sel_m['mae']:.2f} | R2: {sel_m['r2']:.4f} | "
            f"Trained on {int(train_ratio*100)}% of {len(df_feat):,} days"
        )

        yield final_status, pred_fig, fi_fig, metrics_display, pred_display, eda_fig

    except Exception as exc:
        error_msg = f"Error: {str(exc)}"
        logger.exception("Pipeline failed")
        yield error_msg, None, None, None, None, None

# Gradio UI

# CSS: injects custom styles into the Gradio app's HTML
CUSTOM_CSS = """
/* Dark card-style panels */
.gradio-container { background: #0f1117 !important; }
.gr-box { border-radius: 12px !important; border: 1px solid #2a2a3e !important; }
/* Status bar */
#status-bar textarea {
    font-family: 'Courier New', monospace;
    font-size: 13px;
    color: #4CAF50;
    background: #1a1a2e;
}
/* Title */
.app-title { text-align: center; padding: 20px 0; }
"""

# gr.Blocks() keeps the controls and output tabs in one page.
with gr.Blocks(
    title="Stock Price Predictor",
    theme=gr.themes.Base(
        primary_hue="green",
        neutral_hue="slate",
    ),
    css=CUSTOM_CSS,
) as app:

    # Header
    gr.HTML("""
        <div class="app-title">
            <h1 style="font-size:2.2rem; font-weight:800; color:#4CAF50;">
                Stock Price Predictor
            </h1>
            <p style="color:#9E9E9E; font-size:1rem; margin-top:4px;">
                ML-powered short-term forecasting - Linear Regression + Random Forest
            </p>
            <p style="color:#F44336; font-size:0.8rem; margin-top:6px;">
                Educational purposes only. Not financial advice.
            </p>
        </div>
    """)

    # Status bar
    status_bar = gr.Textbox(
        label="System Status",
        value="Ready - configure settings and click Run Analysis",
        interactive=False,
        elem_id="status-bar",
        lines=1,
    )

    # Main layout
    with gr.Row():

        # Left Column: All input controls
        with gr.Column(scale=1, min_width=280):
            gr.Markdown("### Configuration")

            ticker_input = gr.Textbox(
                label="Stock Ticker",
                value=DEFAULT_TICKER,
                placeholder="e.g. AAPL, TSLA, MSFT, GOOGL",
                info="Enter any valid Yahoo Finance ticker symbol",
            )

            # Quick-select buttons for popular tickers
            gr.Markdown("**Quick Select:**")
            with gr.Row():
                btn_aapl = gr.Button("AAPL", size="sm")
                btn_tsla = gr.Button("TSLA", size="sm")
                btn_msft = gr.Button("MSFT", size="sm")
                btn_googl = gr.Button("GOOGL", size="sm")

            # Clicking these buttons sets the ticker_input value
            btn_aapl.click(fn=lambda: "AAPL",  outputs=ticker_input)
            btn_tsla.click(fn=lambda: "TSLA",  outputs=ticker_input)
            btn_msft.click(fn=lambda: "MSFT",  outputs=ticker_input)
            btn_googl.click(fn=lambda: "GOOGL", outputs=ticker_input)

            gr.Markdown("---")

            start_date_input = gr.Textbox(
                label="Start Date (YYYY-MM-DD)",
                value=DEFAULT_START_DATE,
                info="Fetches data from this date forward",
            )
            end_date_input = gr.Textbox(
                label="End Date (YYYY-MM-DD)",
                value=DEFAULT_END_DATE,
            )

            gr.Markdown("---")

            model_dropdown = gr.Dropdown(
                label="Model to Display",
                choices=["Linear Regression", "Random Forest", "Ensemble"],
                value=DEFAULT_MODEL,
                info="All 3 models are always trained; this selects which to show in the prediction chart",
            )

            train_ratio_slider = gr.Slider(
                label="Training Data Ratio",
                minimum=0.60,
                maximum=0.90,
                value=DEFAULT_TRAIN_RATIO,
                step=0.05,
                info="Fraction of data used for training (rest = testing)",
            )

            gr.Markdown("---")
            gr.Markdown("### Random Forest Settings")

            n_trees_slider = gr.Slider(
                label="Number of Trees",
                minimum=50,
                maximum=500,
                value=200,
                step=50,
                info="More trees = more accurate but slower",
            )
            max_depth_slider = gr.Slider(
                label="Max Tree Depth",
                minimum=3,
                maximum=20,
                value=10,
                step=1,
                info="Shallower = less overfitting; deeper = more complex",
            )

            gr.Markdown("---")

            # The main action button
            run_button = gr.Button(
                "Run Analysis",
                variant="primary",
                size="lg",
            )

            gr.Markdown(
                "---\n"
                "**Tips:**\n"
                "- Analysis takes 15-45 seconds\n"
                "- Try different tickers to compare\n"
                "- Hover over charts to inspect values\n"
                "- Download plots using the chart toolbar"
            )

        # Right Column: All outputs in tabs
        with gr.Column(scale=3):
            gr.Markdown("### Analysis Results")

            with gr.Tabs():

                # Tab 1: Price History
                with gr.Tab("Price History"):
                    eda_plot = gr.Plot(
                        label="Historical Price & Volume",
                        show_label=False,
                    )

                # Tab 2: Predictions
                with gr.Tab("Predictions"):
                    pred_plot = gr.Plot(
                        label="Actual vs Predicted",
                        show_label=False,
                    )

                # Tab 3: Next-day forecast
                with gr.Tab("Next-day forecast"):
                    gr.Markdown(
                        "**Prediction for the next trading day** based on today's features.\n\n"
                        "> These are model outputs, not investment advice."
                    )
                    pred_table = gr.Dataframe(
                        label="Next-Day Price Predictions",
                        headers=["Model", "Current Close ($)", "Next Close ($)",
                                 "Change ($)", "Change (%)", "Direction"],
                        wrap=True,
                    )

                # Tab 4: Model Metrics
                with gr.Tab("Model Metrics"):
                    gr.Markdown(
                        "**All three models are evaluated on held-out test data** "
                        "(data the model never saw during training).\n\n"
                        "- **MAE**: Average dollar error\n"
                        "- **RMSE**: Root mean squared error (penalizes large mistakes more)\n"
                        "- **R2**: % of price variance explained (1.0 = perfect)\n"
                        "- **MAPE**: Mean absolute % error\n"
                        "- **Dir Acc**: Did we predict UP vs DOWN correctly?"
                    )
                    metrics_table = gr.Dataframe(
                        label="Evaluation Metrics (Test Set)",
                        headers=["Model", "MAE ($)", "RMSE ($)", "R2", "MAPE (%)", "Dir Acc (%)"],
                        wrap=True,
                    )

                # Tab 5: Feature Importance
                with gr.Tab("Feature Importance"):
                    gr.Markdown(
                        "**Which features does the Random Forest rely on most?**\n\n"
                        "Higher score = feature caused more splits and drove more predictions."
                    )
                    fi_plot = gr.Plot(
                        label="Random Forest Feature Importances",
                        show_label=False,
                    )

    # Footer
    gr.HTML("""
        <div style="text-align:center; margin-top:20px; color:#555; font-size:0.8rem;">
            Built with Python - scikit-learn - yfinance - Gradio - Plotly
            - For educational purposes only
        </div>
    """)

    # Wire controls to the pipeline
    # When run_button is clicked, call run_prediction_pipeline()
    # with the values from all input widgets,
    # and stream outputs one-by-one into the output widgets.
    run_button.click(
        fn=run_prediction_pipeline,
        inputs=[
            ticker_input,
            start_date_input,
            end_date_input,
            model_dropdown,
            train_ratio_slider,
            n_trees_slider,
            max_depth_slider,
        ],
        outputs=[
            status_bar,
            pred_plot,
            fi_plot,
            metrics_table,
            pred_table,
            eda_plot,
        ],
    )

# App entry point
if __name__ == "__main__":
    app.launch(
        server_name="0.0.0.0",   # Listen on all network interfaces
        server_port=7860,         # Default Gradio port
        share=False,              # Set True to get a public gradio.live link
        debug=False,
    )
