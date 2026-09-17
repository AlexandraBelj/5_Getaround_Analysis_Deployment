# Getaround Rental Delay Decision Dashboard
# ---------------------------------------------------------------------------

from pathlib import Path

import pandas as pd
import streamlit as st


# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

LOGO_PATH = (
    PROJECT_ROOT
    / "dashboard"
    / "getaround_logo.png"
)

DELAY_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "get_around_delay_analysis.xlsx"
)


# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Getaround | Rental Delay Decision",
    page_icon=str(LOGO_PATH),
    layout="wide",
)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

@st.cache_data
def load_delay_data():
    return pd.read_excel(
        DELAY_DATA_PATH,
        sheet_name="rentals_data",
    )


df = load_delay_data()


# ---------------------------------------------------------------------------
# Prepare analytical populations
# ---------------------------------------------------------------------------

ended_with_delay = df.loc[
    (df["state"] == "ended")
    & df["delay_at_checkout_in_minutes"].notna()
].copy()

late_rate = (
    (ended_with_delay["delay_at_checkout_in_minutes"] > 0).mean() * 100
)


previous_rentals = df[
    [
        "rental_id",
        "car_id",
        "delay_at_checkout_in_minutes",
    ]
].rename(
    columns={
        "rental_id": "previous_rental_id",
        "car_id": "previous_car_id",
        "delay_at_checkout_in_minutes": "previous_checkout_delay",
    }
)

linked_rentals = df.loc[
    df["previous_ended_rental_id"].notna()
].copy()

linked_rentals["previous_ended_rental_id"] = (
    linked_rentals["previous_ended_rental_id"].astype(int)
)

linked_rentals = linked_rentals.merge(
    previous_rentals,
    left_on="previous_ended_rental_id",
    right_on="previous_rental_id",
    how="left",
)

policy_population = linked_rentals.copy()

problem_population = linked_rentals.loc[
    linked_rentals["previous_checkout_delay"].notna()
].copy()

problem_population["problematic_case"] = (
    problem_population["previous_checkout_delay"]
    > problem_population["time_delta_with_previous_rental_in_minutes"]
)

problem_population["interference_minutes"] = (
    problem_population["previous_checkout_delay"]
    - problem_population["time_delta_with_previous_rental_in_minutes"]
).clip(lower=0)

problematic_cases = problem_population.loc[
    problem_population["problematic_case"]
].copy()

problematic_rate = (
    problem_population["problematic_case"].mean() * 100
)

median_interference = problematic_cases[
    "interference_minutes"
].median()


# ---------------------------------------------------------------------------
# Dashboard header
# ---------------------------------------------------------------------------

st.image(str(LOGO_PATH), width=220)

st.title("Rental Delay Decision Dashboard")

st.markdown(
    """
    ### How much time should Getaround require between two rentals of the same car?

    Late returns can interfere with the next driver's rental. Requiring a
    minimum time gap could reduce that risk — but a larger gap also limits
    how closely owners can schedule consecutive bookings.

    **This dashboard explores that trade-off to support two product decisions:**
    the minimum gap to require and whether the policy should apply to
    **all cars or Connect cars only**.
    """
)

st.divider()


# ---------------------------------------------------------------------------
# Headline business indicators
# ---------------------------------------------------------------------------

st.subheader("Why does this matter?")

st.markdown(
    """
    A late return does **not automatically mean the next driver is affected**.
    Interference occurs only when the previous rental's checkout delay is
    longer than the scheduled gap before the next rental.
    """
)

kpi1, kpi2, kpi3, kpi4 = st.columns(4)

kpi1.metric(
    "Ended rentals returned late",
    f"{late_rate:.1f}%",
)

kpi2.metric(
    "Consecutive rentals analyzed",
    f"{len(problem_population):,}",
)

kpi3.metric(
    "Next-driver interference",
    f"{problematic_rate:.1f}%",
)

kpi4.metric(
    "Median interference",
    f"{median_interference:.1f} min",
)

st.caption(
    "Interference metrics are based on consecutive rentals for which the "
    "previous checkout delay is observed and the planned gap is within "
    "the dataset's documented 12-hour window."
)


# ---------------------------------------------------------------------------
# Interactive policy simulator
# ---------------------------------------------------------------------------

st.divider()

st.subheader("Test a Minimum-Gap Policy")

st.markdown(
    """
    Choose a minimum time between two rentals of the same car and the cars
    covered by the policy. The simulation shows how that rule would have
    changed the historical consecutive-rental configurations in the dataset.
    """
)

control_col1, control_col2 = st.columns([2, 1])

with control_col1:
    threshold = st.select_slider(
        "Minimum required gap",
        options=[0, 30, 60, 90, 120, 180],
        value=60,
        format_func=lambda x: "No minimum" if x == 0 else f"{x} minutes",
    )

with control_col2:
    scope = st.radio(
        "Apply policy to",
        options=["All cars", "Connect cars only"],
        index=1,
    )


# ---------------------------------------------------------------------------
# Apply selected policy
# ---------------------------------------------------------------------------

if scope == "Connect cars only":
    policy_scope = policy_population.loc[
        policy_population["checkin_type"] == "connect"
    ].copy()

    problem_scope = problem_population.loc[
        problem_population["checkin_type"] == "connect"
    ].copy()
else:
    policy_scope = policy_population.copy()
    problem_scope = problem_population.copy()


restricted = (
    policy_scope["time_delta_with_previous_rental_in_minutes"]
    < threshold
)

problematic = problem_scope["problematic_case"]

potentially_prevented = (
    problematic
    & (
        problem_scope["time_delta_with_previous_rental_in_minutes"]
        < threshold
    )
)

restricted_count = int(restricted.sum())
restricted_pct = restricted.mean() * 100

problematic_count = int(problematic.sum())
prevented_count = int(potentially_prevented.sum())

prevented_pct = (
    prevented_count / problematic_count * 100
    if problematic_count > 0
    else 0
)


# ---------------------------------------------------------------------------
# Selected-policy results
# ---------------------------------------------------------------------------

st.markdown("### What would this policy change?")

result_col1, result_col2, result_col3 = st.columns(3)

result_col1.metric(
    "Bookings requiring different timing",
    f"{restricted_pct:.1f}%",
    help=(
        "Share of observed consecutive-rental configurations whose planned "
        "gap was shorter than the selected minimum."
    ),
)

result_col2.metric(
    "Interference cases potentially prevented",
    f"{prevented_pct:.1f}%",
    help=(
        "Share of historical problematic cases occurring with a gap shorter "
        "than the selected minimum."
    ),
)

result_col3.metric(
    "Problematic cases potentially addressed",
    f"{prevented_count} / {problematic_count}",
    help=(
        "Number of observed problematic cases potentially addressed by the "
        "selected policy."
    ),
)


if threshold == 0:
    st.info(
        "With no minimum gap, historical booking configurations remain "
        "unrestricted, but the policy does not address any observed "
        "next-driver interference."
    )
else:
    st.info(
        f"A **{threshold}-minute minimum gap** for **{scope.lower()}** would "
        f"have required different timing for **{restricted_count:,}** "
        f"consecutive bookings (**{restricted_pct:.1f}%**) while potentially "
        f"addressing **{prevented_count} of {problematic_count}** observed "
        f"interference cases (**{prevented_pct:.1f}%**)."
    )

st.caption(
    "A booking requiring different timing is not necessarily a lost booking. "
    "It could be rescheduled or fulfilled with another vehicle."
)


# ---------------------------------------------------------------------------
# Compare policy options
# ---------------------------------------------------------------------------

st.divider()

st.subheader("Compare Policy Options")

st.markdown(
    """
    Increasing the minimum gap protects more next drivers, but also requires
    more consecutive bookings to be scheduled differently. Compare the
    alternatives below for the currently selected policy scope.
    """
)

threshold_options = [0, 30, 60, 90, 120, 180]

comparison_rows = []

for threshold_option in threshold_options:

    restricted_option = (
        policy_scope["time_delta_with_previous_rental_in_minutes"]
        < threshold_option
    )

    prevented_option = (
        problem_scope["problematic_case"]
        & (
            problem_scope["time_delta_with_previous_rental_in_minutes"]
            < threshold_option
        )
    )

    restricted_option_pct = restricted_option.mean() * 100

    prevented_option_count = int(prevented_option.sum())

    prevented_option_pct = (
        prevented_option_count / problematic_count * 100
        if problematic_count > 0
        else 0
    )

    comparison_rows.append(
        {
            "Minimum gap": threshold_option,
            "Bookings requiring different timing (%)": restricted_option_pct,
            "Interference potentially prevented (%)": prevented_option_pct,
            "Problematic cases potentially addressed": (
                f"{prevented_option_count} / {problematic_count}"
            ),
        }
    )

comparison_df = pd.DataFrame(comparison_rows)


# ---------------------------------------------------------------------------
# Policy trade-off chart
# ---------------------------------------------------------------------------

st.markdown("### Protection vs. Scheduling Restriction")

chart_df = comparison_df.set_index("Minimum gap")[
    [
        "Bookings requiring different timing (%)",
        "Interference potentially prevented (%)",
    ]
]

st.line_chart(
    chart_df,
    x_label="Minimum required gap (minutes)",
    y_label="Share (%)",
)

st.caption(
    "The benefit of larger minimum gaps gradually shows diminishing returns: "
    "additional protection comes at the cost of restricting more historical "
    "consecutive-booking configurations."
)


# ---------------------------------------------------------------------------
# Policy comparison table
# ---------------------------------------------------------------------------

st.markdown("### Scenario Comparison")

display_comparison = comparison_df.copy()

display_comparison["Minimum gap"] = display_comparison[
    "Minimum gap"
].apply(
    lambda x: "No minimum" if x == 0 else f"{x} min"
)

display_comparison[
    "Bookings requiring different timing (%)"
] = display_comparison[
    "Bookings requiring different timing (%)"
].map(lambda x: f"{x:.1f}%")

display_comparison[
    "Interference potentially prevented (%)"
] = display_comparison[
    "Interference potentially prevented (%)"
].map(lambda x: f"{x:.1f}%")

st.dataframe(
    display_comparison,
    hide_index=True,
    use_container_width=True,
)


# ---------------------------------------------------------------------------
# Recommended scenario metrics
# ---------------------------------------------------------------------------

RECOMMENDED_THRESHOLD = 60

recommended_policy_scope = policy_population.loc[
    policy_population["checkin_type"] == "connect"
].copy()

recommended_problem_scope = problem_population.loc[
    problem_population["checkin_type"] == "connect"
].copy()

recommended_restricted = (
    recommended_policy_scope["time_delta_with_previous_rental_in_minutes"]
    < RECOMMENDED_THRESHOLD
)

recommended_problematic = recommended_problem_scope["problematic_case"]

recommended_prevented = (
    recommended_problematic
    & (
        recommended_problem_scope[
            "time_delta_with_previous_rental_in_minutes"
        ]
        < RECOMMENDED_THRESHOLD
    )
)

recommended_restricted_pct = recommended_restricted.mean() * 100

recommended_problematic_count = int(recommended_problematic.sum())
recommended_prevented_count = int(recommended_prevented.sum())

recommended_prevented_pct = (
    recommended_prevented_count
    / recommended_problematic_count
    * 100
)


# ---------------------------------------------------------------------------
# Product recommendation
# ---------------------------------------------------------------------------

st.divider()

st.subheader("Recommended Starting Point")

st.success(
    f"""
    ### Start by testing a {RECOMMENDED_THRESHOLD}-minute minimum gap on Connect cars

    In the historical data, this scenario would have required different
    timing for about **{recommended_restricted_pct:.1f}% of consecutive
    Connect bookings**, while potentially addressing about
    **{recommended_prevented_pct:.1f}% of observed interference cases**
    in that scope.

    This provides a practical starting point for testing the policy without
    immediately applying it to the entire fleet.
    """
)

rec_col1, rec_col2, rec_col3 = st.columns(3)

rec_col1.metric(
    "Proposed minimum gap",
    f"{RECOMMENDED_THRESHOLD} min",
)

rec_col2.metric(
    "Bookings requiring different timing",
    f"{recommended_restricted_pct:.1f}%",
)

rec_col3.metric(
    "Interference potentially addressed",
    f"{recommended_prevented_pct:.1f}%",
)

st.markdown(
    """
    **Why start here?**

    A **30-minute gap** has a smaller scheduling impact, but leaves more than
    40% of the observed problematic Connect cases unaddressed.

    Moving to **60 minutes** increases the share of potentially addressed
    interference cases to about 70%, while keeping the share of historical
    consecutive bookings requiring different timing near 22%.

    Larger thresholds provide additional protection, but increasingly
    restrict scheduling flexibility. For example, a **90-minute gap** would
    require different timing for about 32% of consecutive Connect bookings.
    """
)


# ---------------------------------------------------------------------------
# Next-driver impact
# ---------------------------------------------------------------------------

st.markdown("### What does an interference case look like?")

st.markdown(
    """
    A late checkout becomes a next-driver problem only when the delay is
    longer than the time scheduled before the next rental.

    **Example:** if a car is returned **90 minutes late** and the next rental
    was scheduled **60 minutes later**, the next driver faces
    **30 minutes of potential interference**.
    """
)

impact_col1, impact_col2 = st.columns([1, 2])

with impact_col1:

    st.metric(
        "Median observed interference",
        f"{median_interference:.1f} min",
    )

    st.metric(
        "Problematic cases analyzed",
        f"{len(problematic_cases):,}",
    )

with impact_col2:
    severity_data = pd.DataFrame(
        {
            "Severity": [
                "≤15 min",
                "16–30 min",
                "31–60 min",
                "61–120 min",
                ">120 min",
            ],
            "Share of problematic cases (%)": [
                (
                    problematic_cases["interference_minutes"] <= 15
                ).mean() * 100,
                (
                    (problematic_cases["interference_minutes"] > 15)
                    & (problematic_cases["interference_minutes"] <= 30)
                ).mean() * 100,
                (
                    (problematic_cases["interference_minutes"] > 30)
                    & (problematic_cases["interference_minutes"] <= 60)
                ).mean() * 100,
                (
                    (problematic_cases["interference_minutes"] > 60)
                    & (problematic_cases["interference_minutes"] <= 120)
                ).mean() * 100,
                (
                    problematic_cases["interference_minutes"] > 120
                ).mean() * 100,
            ],
        }
    )

    st.bar_chart(
        severity_data,
        x="Severity",
        y="Share of problematic cases (%)",
        x_label="Interference duration",
        y_label="Share of problematic cases (%)",
    )

st.caption(
    "Just over half of observed interference cases lasted 30 minutes or less, "
    "while a smaller group involved substantially longer disruption."
)

st.markdown("### What should Getaround measure next?")

st.markdown(
    """
    A historical simulation cannot show how owners and drivers would react
    after the policy is introduced. An initial rollout should therefore track:

    - **Booking conversion** — do fewer users complete a booking?
    - **Rescheduling** — how often can restricted bookings move to another time?
    - **Cancellations** — does the policy increase abandoned rentals?
    - **Owner revenue** — does reduced scheduling flexibility affect earnings?
    - **Next-driver incidents** — does interference actually decline after rollout?

    These outcomes would allow Getaround to evaluate whether the operational
    benefit of the minimum gap outweighs its effect on marketplace activity.
    """
)

st.caption(
    "The 60-minute Connect-only scenario is a product-testing recommendation, "
    "not a claim that it is the mathematically optimal policy."
)


# ---------------------------------------------------------------------------
# Data context and limitations
# ---------------------------------------------------------------------------

st.divider()

st.subheader("About the Data & Limitations")

with st.expander("View methodology and data limitations"):
    st.markdown(
        f"""
        **Dataset coverage**

        - **{len(df):,} rentals** across **{df['car_id'].nunique():,} cars**
        - **{len(policy_population):,} consecutive-rental situations** observed
          within the documented 12-hour window
        - **{len(problem_population):,} consecutive situations** with an
          observed previous checkout delay available for interference analysis

        **Important limitations**

        **Missing checkout delays:** some completed rentals do not have an
        observed checkout delay. These values are not imputed, so
        interference analysis uses only cases where the previous rental's
        delay is available.

        **12-hour observation window:** the dataset identifies the previous
        ended rental only when the planned gap is below 12 hours. The analysis
        therefore does not represent every historical sequence of rentals.

        **Historical simulation, not causal evidence:** a case is considered
        potentially addressed when its historical booking configuration would
        violate the proposed minimum-gap rule. This does not prove that the
        same incident would have been prevented after implementation.

        **Restricted does not mean lost:** bookings requiring different timing
        could potentially be rescheduled or fulfilled with another vehicle.

        **Revenue cannot be measured directly:** the separate pricing dataset
        does not share a `rental_id` or `car_id` with the delay dataset.
        Therefore, the exact historical owner revenue affected by a
        minimum-gap policy cannot be calculated from the available data.
        """
    )

st.caption(
    "Getaround Rental Delay Decision Dashboard · "
    "Historical decision-support analysis"
)