#!/usr/bin/env python
# coding: utf-8

"""
work_on_this.py
================
Supply Chain AI Agent — core logic module.
Import `generate_data` and `build_graph` into app.py.
"""

# ==============================
# IMPORTS
# ==============================

import os
import traceback
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv

from typing import Annotated
from typing_extensions import TypedDict

from langchain_core.tools import tool
from langchain_core.messages import AnyMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END, add_messages
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv()

# ==============================
# CONFIGURATION
# ==============================

NUM_SKUS = 500
NUM_STORES = 10
DAYS_HISTORY = 180
START_DATE = datetime.today() - timedelta(days=DAYS_HISTORY)
np.random.seed(42)


# ==============================
# LANGGRAPH STATE SCHEMA
# ==============================

class State(TypedDict, total=False):
    sc_state: object
    messages: Annotated[list[AnyMessage], add_messages]


# ==============================
# SUPPLY CHAIN STATE CLASS
# ==============================

class SupplyChainState:
    def __init__(
        self,
        sku_master,
        central_inventory,
        store_inventory,
        demand_history,
        open_po,
        inventory_aging,
        transfer_leadtime,
        cost_parameters,
    ):
        self.sku_master = sku_master.copy()
        self.central_inventory = central_inventory.copy()
        self.store_inventory = store_inventory.copy()
        self.demand_history = demand_history.copy()
        self.open_po = open_po.copy()
        self.inventory_aging = inventory_aging.copy()
        self.transfer_leadtime = transfer_leadtime.copy()
        self.cost_parameters = cost_parameters.copy()
        self.action_log = []
        self.last_updated = datetime.now()


# ==============================
# DATA GENERATION
# ==============================

def generate_data() -> SupplyChainState:
    """Generate all synthetic supply chain dataframes. Returns a SupplyChainState."""

    sku_ids = [f"SKU_{i:04d}" for i in range(1, NUM_SKUS + 1)]
    abc_classes = (
        ["A"] * int(0.2 * NUM_SKUS)
        + ["B"] * int(0.3 * NUM_SKUS)
        + ["C"] * int(0.5 * NUM_SKUS)
    )
    np.random.shuffle(abc_classes)

    sku_master_df = pd.DataFrame({"sku_id": sku_ids, "abc_class": abc_classes})

    def assign_parameters(row):
        if row["abc_class"] == "A":
            return pd.Series({
                "unit_cost": np.random.uniform(50, 150),
                "selling_price": np.random.uniform(120, 300),
                "lead_time_days": np.random.randint(7, 15),
                "moq": np.random.randint(100, 300),
                "safety_stock": np.random.randint(200, 500),
            })
        elif row["abc_class"] == "B":
            return pd.Series({
                "unit_cost": np.random.uniform(20, 80),
                "selling_price": np.random.uniform(80, 150),
                "lead_time_days": np.random.randint(10, 20),
                "moq": np.random.randint(50, 200),
                "safety_stock": np.random.randint(100, 300),
            })
        else:
            return pd.Series({
                "unit_cost": np.random.uniform(5, 30),
                "selling_price": np.random.uniform(20, 70),
                "lead_time_days": np.random.randint(15, 30),
                "moq": np.random.randint(20, 100),
                "safety_stock": np.random.randint(20, 100),
            })

    sku_master_df = pd.concat(
        [sku_master_df, sku_master_df.apply(assign_parameters, axis=1)], axis=1
    )
    sku_master_df["reorder_point"] = sku_master_df["safety_stock"] * 1.2
    sku_master_df["danger_level"] = sku_master_df["safety_stock"] * 0.5

    # --- Demand History ---
    demand_records = []
    for _, row in sku_master_df.iterrows():
        sku, abc = row["sku_id"], row["abc_class"]
        if abc == "A":
            base_demand, variability = np.random.uniform(30, 60), 0.1
        elif abc == "B":
            base_demand, variability = np.random.uniform(10, 30), 0.3
        else:
            base_demand, variability = np.random.uniform(1, 10), 0.6
        for day in range(DAYS_HISTORY):
            date = START_DATE + timedelta(days=day)
            demand = max(0, np.random.normal(base_demand, base_demand * variability))
            demand_records.append([sku, date, round(demand)])
    demand_history_df = pd.DataFrame(
        demand_records, columns=["sku_id", "date", "daily_demand"]
    )

    # --- Central Inventory ---
    inventory_central_df = sku_master_df[["sku_id"]].copy()
    inventory_central_df["current_stock"] = np.random.randint(0, 2000, size=NUM_SKUS)
    zero_idx = np.random.choice(NUM_SKUS, 15, replace=False)
    inventory_central_df.loc[zero_idx, "current_stock"] = 0
    inventory_central_df["in_transit_stock"] = np.random.randint(0, 500, size=NUM_SKUS)
    inventory_central_df["last_updated"] = datetime.today()

    # --- Store Inventory ---
    store_ids = [f"STORE_{i}" for i in range(1, NUM_STORES + 1)]
    store_inventory_records = [
        [sku, store, np.random.randint(0, 500)]
        for sku in sku_ids
        for store in store_ids
    ]
    inventory_store_df = pd.DataFrame(
        store_inventory_records, columns=["sku_id", "store_id", "current_stock"]
    )

    # --- Open POs ---
    po_records = [
        [
            f"PO_{i}",
            np.random.choice(sku_ids),
            np.random.randint(50, 500),
            datetime.today() + timedelta(days=np.random.randint(5, 25)),
        ]
        for i in range(200)
    ]
    open_po_df = pd.DataFrame(
        po_records,
        columns=["po_id", "sku_id", "ordered_qty", "expected_delivery_date"],
    )

    # --- Inventory Aging ---
    aging_records = [
        [sku, f"{sku}_B{batch}", np.random.randint(10, 300), np.random.randint(0, 365)]
        for sku in sku_ids
        for batch in range(np.random.randint(1, 4))
    ]
    inventory_aging_df = pd.DataFrame(
        aging_records, columns=["sku_id", "batch_id", "qty", "age_days"]
    )

    # --- Transfer Lead Times ---
    locations = ["CENTRAL"] + store_ids
    transfer_records = [
        [fl, tl, np.random.randint(1, 5), np.random.uniform(1, 10)]
        for fl in locations
        for tl in locations
        if fl != tl
    ]
    transfer_leadtime_df = pd.DataFrame(
        transfer_records,
        columns=[
            "from_location",
            "to_location",
            "transfer_lead_time_days",
            "transfer_cost_per_unit",
        ],
    )

    # --- Cost Parameters ---
    cost_parameters_df = sku_master_df[["sku_id", "abc_class", "unit_cost"]].copy()
    cost_parameters_df["holding_cost_per_unit_per_day"] = (
        cost_parameters_df["unit_cost"] * 0.0005
    )
    cost_parameters_df["stockout_cost_per_unit"] = cost_parameters_df["unit_cost"] * 2
    cost_parameters_df["service_level_target"] = cost_parameters_df["abc_class"].apply(
        lambda abc: {"A": 0.98, "B": 0.95}.get(abc, 0.90)
    )

    return SupplyChainState(
        sku_master=sku_master_df,
        central_inventory=inventory_central_df,
        store_inventory=inventory_store_df,
        demand_history=demand_history_df,
        open_po=open_po_df,
        inventory_aging=inventory_aging_df,
        transfer_leadtime=transfer_leadtime_df,
        cost_parameters=cost_parameters_df,
    )


# ==============================
# TOOLS FACTORY
# Builds all tools scoped to a given sc_state instance (no globals needed)
# ==============================

def build_tools(sc_state: SupplyChainState):
    """Return list of all LangChain tools bound to the provided sc_state."""

    @tool
    def zero_stock_node() -> dict:
        """Return count and list of SKUs with zero stock in central warehouse."""
        zero_skus = sc_state.central_inventory[
            sc_state.central_inventory["current_stock"] <= 0
        ]["sku_id"].tolist()
        return {"zero_stock_count": len(zero_skus), "zero_stock_skus": zero_skus}

    @tool
    def reorder_alert_tool() -> dict:
        """Identify SKUs below reorder point with no open purchase order — needs immediate PO creation."""
        merged = sc_state.central_inventory.merge(
            sc_state.sku_master[["sku_id", "reorder_point", "abc_class"]], on="sku_id"
        )
        below = merged[merged["current_stock"] < merged["reorder_point"]][
            ["sku_id", "current_stock", "reorder_point", "abc_class"]
        ]
        skus_with_po = sc_state.open_po["sku_id"].unique().tolist()
        no_po = below[~below["sku_id"].isin(skus_with_po)]
        return {
            "total_below_reorder": len(below),
            "skus_with_no_po": len(no_po),
            "details": no_po.sort_values("abc_class").to_dict(orient="records"),
        }

    @tool
    def danger_level_breach_tool() -> dict:
        """Identify SKUs that have breached the danger level threshold — emergency replenishment needed."""
        merged = sc_state.central_inventory.merge(
            sc_state.sku_master[["sku_id", "danger_level", "abc_class", "lead_time_days"]],
            on="sku_id",
        )
        breached = merged[merged["current_stock"] < merged["danger_level"]][
            ["sku_id", "current_stock", "danger_level", "abc_class", "lead_time_days"]
        ]
        return {
            "danger_level_breach_count": len(breached),
            "details": breached.sort_values("abc_class").to_dict(orient="records"),
        }

    @tool
    def stockout_risk_tool() -> dict:
        """
        Calculate days of stock remaining per SKU using avg daily demand (last 30 days).
        Flag SKUs that will hit zero before replenishment arrives within the lead time window.
        """
        rd = sc_state.demand_history.copy()
        rd["date"] = pd.to_datetime(rd["date"])
        cutoff = rd["date"].max() - pd.Timedelta(days=30)
        avg_demand = (
            rd[rd["date"] >= cutoff]
            .groupby("sku_id")["daily_demand"]
            .mean()
            .reset_index()
            .rename(columns={"daily_demand": "avg_daily_demand"})
        )
        merged = sc_state.central_inventory.merge(avg_demand, on="sku_id").merge(
            sc_state.sku_master[["sku_id", "lead_time_days", "abc_class"]], on="sku_id"
        )
        merged["days_of_cover"] = merged.apply(
            lambda r: round(r["current_stock"] / r["avg_daily_demand"], 1)
            if r["avg_daily_demand"] > 0
            else 9999,
            axis=1,
        )
        at_risk = merged[merged["days_of_cover"] < merged["lead_time_days"]][
            ["sku_id", "current_stock", "avg_daily_demand", "days_of_cover", "lead_time_days", "abc_class"]
        ]
        return {
            "at_risk_count": len(at_risk),
            "details": at_risk.sort_values("days_of_cover").to_dict(orient="records"),
        }

    @tool
    def po_coverage_tool() -> dict:
        """Check if open POs cover the gap for SKUs below reorder point. Returns covered vs uncovered split."""
        merged = sc_state.central_inventory.merge(
            sc_state.sku_master[["sku_id", "reorder_point", "safety_stock", "abc_class"]],
            on="sku_id",
        )
        below = merged[merged["current_stock"] < merged["reorder_point"]]
        po_sum = (
            sc_state.open_po.groupby("sku_id")["ordered_qty"]
            .sum()
            .reset_index()
            .rename(columns={"ordered_qty": "total_po_qty"})
        )
        cov = below.merge(po_sum, on="sku_id", how="left")
        cov["total_po_qty"] = cov["total_po_qty"].fillna(0)
        cov["projected_stock"] = cov["current_stock"] + cov["total_po_qty"]
        cov["is_covered"] = cov["projected_stock"] >= cov["safety_stock"]
        covered = cov[cov["is_covered"]][["sku_id", "current_stock", "total_po_qty", "projected_stock"]]
        not_covered = cov[~cov["is_covered"]][
            ["sku_id", "current_stock", "total_po_qty", "projected_stock", "abc_class"]
        ]
        return {
            "already_covered_count": len(covered),
            "needs_new_po_count": len(not_covered),
            "needs_new_po": not_covered.sort_values("abc_class").to_dict(orient="records"),
        }

    @tool
    def recommended_order_qty_tool() -> dict:
        """
        Calculate recommended order quantity per SKU.
        Formula: demand over lead time + safety stock − (current + in_transit + open POs).
        Respects MOQ. Only returns SKUs that actually need ordering.
        """
        avg_demand = (
            sc_state.demand_history.groupby("sku_id")["daily_demand"]
            .mean()
            .reset_index()
            .rename(columns={"daily_demand": "avg_daily_demand"})
        )
        merged = sc_state.central_inventory.merge(avg_demand, on="sku_id").merge(
            sc_state.sku_master[
                ["sku_id", "lead_time_days", "safety_stock", "moq", "abc_class"]
            ],
            on="sku_id",
        )
        po_sum = (
            sc_state.open_po.groupby("sku_id")["ordered_qty"]
            .sum()
            .reset_index()
            .rename(columns={"ordered_qty": "total_po_qty"})
        )
        merged = merged.merge(po_sum, on="sku_id", how="left")
        merged["total_po_qty"] = merged["total_po_qty"].fillna(0)
        merged["required_stock"] = (
            merged["avg_daily_demand"] * merged["lead_time_days"] + merged["safety_stock"]
        )
        merged["available_stock"] = (
            merged["current_stock"] + merged["in_transit_stock"] + merged["total_po_qty"]
        )
        merged["raw_order_qty"] = merged["required_stock"] - merged["available_stock"]
        needs = merged[merged["raw_order_qty"] > 0].copy()
        needs["recommended_qty"] = needs.apply(
            lambda r: max(r["raw_order_qty"], r["moq"]), axis=1
        ).round(0)
        return {
            "skus_needing_order": len(needs),
            "details": needs[
                [
                    "sku_id", "abc_class", "current_stock", "available_stock",
                    "required_stock", "recommended_qty", "moq",
                ]
            ]
            .sort_values("abc_class")
            .to_dict(orient="records"),
        }

    @tool
    def store_transfer_opportunity_tool() -> dict:
        """Find SKUs where central is below reorder point but stores hold excess — transfer instead of new PO."""
        low_central = sc_state.central_inventory.merge(
            sc_state.sku_master[["sku_id", "reorder_point", "safety_stock"]], on="sku_id"
        )
        low_central = low_central[
            low_central["current_stock"] < low_central["reorder_point"]
        ][["sku_id", "current_stock", "safety_stock"]]
        store_stock = (
            sc_state.store_inventory.groupby("sku_id")["current_stock"]
            .sum()
            .reset_index()
            .rename(columns={"current_stock": "total_store_stock"})
        )
        opp = low_central.merge(store_stock, on="sku_id")
        opp["store_excess"] = opp["total_store_stock"] - opp["safety_stock"]
        opp = opp[opp["store_excess"] > 0]
        return {
            "transfer_opportunities_count": len(opp),
            "details": opp[
                ["sku_id", "current_stock", "total_store_stock", "store_excess", "safety_stock"]
            ]
            .sort_values("store_excess", ascending=False)
            .to_dict(orient="records"),
        }

    @tool
    def store_stockout_risk_tool() -> dict:
        """Identify store-SKU combos critically low on stock and check if central can fulfill replenishment."""
        avg_demand = (
            sc_state.demand_history.groupby("sku_id")["daily_demand"]
            .mean()
            .reset_index()
            .rename(columns={"daily_demand": "avg_daily_demand"})
        )
        sm = sc_state.store_inventory.merge(avg_demand, on="sku_id")
        ct = sc_state.transfer_leadtime[
            sc_state.transfer_leadtime["from_location"] == "CENTRAL"
        ][["to_location", "transfer_lead_time_days"]]
        sm = sm.merge(ct, left_on="store_id", right_on="to_location", how="left")
        sm["transfer_lead_time_days"] = sm["transfer_lead_time_days"].fillna(3)
        sm["days_of_cover"] = sm.apply(
            lambda r: round(r["current_stock"] / r["avg_daily_demand"], 1)
            if r["avg_daily_demand"] > 0
            else 9999,
            axis=1,
        )
        at_risk = sm[sm["days_of_cover"] < sm["transfer_lead_time_days"]][
            ["sku_id", "store_id", "current_stock", "avg_daily_demand", "days_of_cover", "transfer_lead_time_days"]
        ]
        central_stock = sc_state.central_inventory[["sku_id", "current_stock"]].rename(
            columns={"current_stock": "central_stock"}
        )
        at_risk = at_risk.merge(central_stock, on="sku_id", how="left")
        at_risk["can_central_fulfill"] = at_risk["central_stock"] > 0
        return {
            "at_risk_store_sku_count": len(at_risk),
            "details": at_risk.sort_values("days_of_cover").to_dict(orient="records"),
        }

    @tool
    def dead_stock_tool(age_threshold_days: int = 180) -> dict:
        """
        Flag SKUs with batches older than age_threshold_days (default 180).
        Capital tied up — actionable: markdown, liquidate, or return to vendor.
        """
        aged = sc_state.inventory_aging[
            sc_state.inventory_aging["age_days"] >= age_threshold_days
        ]
        summary = (
            aged.groupby("sku_id")
            .agg(
                total_aged_qty=("qty", "sum"),
                max_age_days=("age_days", "max"),
                num_batches=("batch_id", "count"),
            )
            .reset_index()
        )
        summary = summary.merge(
            sc_state.sku_master[["sku_id", "abc_class", "unit_cost"]], on="sku_id"
        )
        summary["capital_tied_up"] = summary["total_aged_qty"] * summary["unit_cost"]
        return {
            "dead_stock_sku_count": len(summary),
            "total_capital_tied_up": round(summary["capital_tied_up"].sum(), 2),
            "details": summary.sort_values("capital_tied_up", ascending=False).to_dict(
                orient="records"
            ),
        }

    @tool
    def overstock_tool(cover_days: int = 90) -> dict:
        """
        Identify SKUs where current stock exceeds cover_days of demand (default 90).
        Highlights monthly holding cost waste from carrying excess inventory.
        """
        avg_demand = (
            sc_state.demand_history.groupby("sku_id")["daily_demand"]
            .mean()
            .reset_index()
            .rename(columns={"daily_demand": "avg_daily_demand"})
        )
        merged = (
            sc_state.central_inventory.merge(avg_demand, on="sku_id")
            .merge(sc_state.sku_master[["sku_id", "abc_class", "unit_cost"]], on="sku_id")
            .merge(
                sc_state.cost_parameters[["sku_id", "holding_cost_per_unit_per_day"]],
                on="sku_id",
            )
        )
        merged["max_recommended_stock"] = merged["avg_daily_demand"] * cover_days
        merged["excess_qty"] = merged["current_stock"] - merged["max_recommended_stock"]
        over = merged[merged["excess_qty"] > 0].copy()
        over["daily_holding_cost"] = over["excess_qty"] * over["holding_cost_per_unit_per_day"]
        over["monthly_holding_cost"] = over["daily_holding_cost"] * 30
        return {
            "overstock_sku_count": len(over),
            "total_monthly_holding_cost_waste": round(over["monthly_holding_cost"].sum(), 2),
            "details": over[
                [
                    "sku_id", "abc_class", "current_stock", "max_recommended_stock",
                    "excess_qty", "monthly_holding_cost",
                ]
            ]
            .sort_values("monthly_holding_cost", ascending=False)
            .to_dict(orient="records"),
        }

    @tool
    def stockout_cost_estimator_tool() -> dict:
        """
        For zero/near-zero stock SKUs, estimate daily and weekly revenue loss.
        Prioritises which stockouts to fix first by financial impact.
        """
        avg_demand = (
            sc_state.demand_history.groupby("sku_id")["daily_demand"]
            .mean()
            .reset_index()
            .rename(columns={"daily_demand": "avg_daily_demand"})
        )
        merged = (
            sc_state.central_inventory.merge(
                sc_state.sku_master[["sku_id", "safety_stock", "selling_price", "abc_class"]],
                on="sku_id",
            )
            .merge(sc_state.cost_parameters[["sku_id", "stockout_cost_per_unit"]], on="sku_id")
            .merge(avg_demand, on="sku_id")
        )
        at_risk = merged[merged["current_stock"] <= merged["safety_stock"]].copy()
        at_risk["daily_revenue_at_risk"] = at_risk["avg_daily_demand"] * at_risk["selling_price"]
        at_risk["daily_stockout_cost"] = (
            at_risk["avg_daily_demand"] * at_risk["stockout_cost_per_unit"]
        )
        at_risk["weekly_revenue_at_risk"] = at_risk["daily_revenue_at_risk"] * 7
        return {
            "at_risk_sku_count": len(at_risk),
            "total_daily_revenue_at_risk": round(at_risk["daily_revenue_at_risk"].sum(), 2),
            "total_weekly_revenue_at_risk": round(at_risk["weekly_revenue_at_risk"].sum(), 2),
            "details": at_risk[
                [
                    "sku_id", "abc_class", "current_stock", "safety_stock",
                    "avg_daily_demand", "daily_revenue_at_risk", "weekly_revenue_at_risk",
                ]
            ]
            .sort_values("daily_revenue_at_risk", ascending=False)
            .to_dict(orient="records"),
        }

    @tool
    def general_analytics_tool(code: str) -> str:
        """
        General-purpose analytics tool. Executes Python/pandas code against sc_state.
        Available dataframes via sc_state:
          sku_master, central_inventory, store_inventory, demand_history,
          open_po, inventory_aging, transfer_leadtime, cost_parameters.
        Code MUST assign the final answer to a variable named `result`.
        """
        exec_globals = {
            "sc_state": sc_state,
            "pd": pd,
            "np": np,
            "datetime": datetime,
        }
        exec_locals = {}
        try:
            exec(code, exec_globals, exec_locals)
            result = exec_locals.get("result")
            if result is None:
                return "Code ran but no `result` variable was set."
            return str(result)
        except Exception:
            return f"Error:\n{traceback.format_exc()}"

    return [
        zero_stock_node,
        reorder_alert_tool,
        danger_level_breach_tool,
        stockout_risk_tool,
        po_coverage_tool,
        recommended_order_qty_tool,
        store_transfer_opportunity_tool,
        store_stockout_risk_tool,
        dead_stock_tool,
        overstock_tool,
        stockout_cost_estimator_tool,
        general_analytics_tool,
    ]


# ==============================
# GRAPH BUILDER
# ==============================

def build_graph(sc_state: SupplyChainState):
    """
    Compile the LangGraph agent.
    Returns: (compiled_graph, all_tools)
    """
    all_tools = build_tools(sc_state)
    from langchain_groq import ChatGroq
    # LLM with timeout + retry (from latest project code)
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.2,timeout=45, max_retries=3)
    llm_with_tools = llm.bind_tools(all_tools)

    def tool_calling_llm(state: State):
        messages = state.get("messages", [])
        sc = state.get("sc_state")
        if not messages:
            return {"messages": [], "sc_state": sc}
        response = llm_with_tools.invoke(messages)
        return {"messages": [response], "sc_state": sc}

    builder = StateGraph(State)
    builder.add_node("tool_calling_llm", tool_calling_llm)
    builder.add_node("tools", ToolNode(all_tools))
    builder.add_edge(START, "tool_calling_llm")
    builder.add_conditional_edges("tool_calling_llm", tools_condition)
    builder.add_edge("tools", "tool_calling_llm")

    return builder.compile(), all_tools
