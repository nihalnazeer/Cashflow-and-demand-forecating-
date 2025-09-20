"use client";

import { useEffect } from "react";
import { useDashboardStore } from "../../lib/store";

// --- All of your component imports ---
import KPIHeader from "../../components/shared/KPIHeader";
import DashboardFilters from "../../components/shared/DashboardFilters";
import HistoricalTrendChart from "../../components/analysis/HistoricalTrendChart";
import PerformanceTable from "../../components/analysis/PerformanceTable";
import ForecastTable from "../../components/predictive/ForecastTable";
import InsightPanel from "../../components/predictive/InsightPanel";
// ... and any other components you are using

export default function DashboardPage() {
  // ✨ Get the new master fetch action from the store
  const fetchAllInitialData = useDashboardStore((state) => state.fetchAllInitialData);

  // ✨ Use useEffect to call the master action once when the page loads
  useEffect(() => {
    // This single function will now fetch the data for the chart, KPIs, and table
    fetchAllInitialData();
  }, [fetchAllInitialData]);

  return (
    <div className="flex h-screen bg-gradient-to-br from-gray-50 to-white text-foreground">
      {/* ... your sidebar JSX ... */}
      
      <div className="flex-1 flex flex-col overflow-hidden">
        {/* ... your header JSX ... */}
        
        <main className="flex-1 overflow-y-auto">
          {/* Main Content Container with improved spacing */}
          <div className="p-6 space-y-8 max-w-[1600px] mx-auto">
            
            {/* KPI Section */}
            <section className="space-y-2">
              <div className="flex items-center gap-3 mb-4">
                <div className="w-1 h-6 bg-gradient-to-b from-primary to-primary/60 rounded-full"></div>
                <h2 className="text-lg font-semibold text-foreground">Key Metrics</h2>
              </div>
              <KPIHeader />
            </section>

            {/* Filters Section */}
            <section className="space-y-2">
              <div className="flex items-center gap-3 mb-4">
                <div className="w-1 h-6 bg-gradient-to-b from-blue-500 to-blue-400 rounded-full"></div>
                <h2 className="text-lg font-semibold text-foreground">Filters & Controls</h2>
              </div>
              <DashboardFilters />
            </section>

            {/* Analytics Section */}
            <section className="space-y-6">
              <div className="flex items-center gap-3 mb-6">
                <div className="w-1 h-6 bg-gradient-to-b from-emerald-500 to-emerald-400 rounded-full"></div>
                <h2 className="text-lg font-semibold text-foreground">Performance Analytics</h2>
              </div>
              
              {/* Primary Analytics Grid */}
              <div className="grid grid-cols-1 xl:grid-cols-5 gap-6">
                {/* Chart takes more space on larger screens */}
                <div className="xl:col-span-3">
                  <HistoricalTrendChart />
                </div>
                {/* Table takes remaining space */}
                <div className="xl:col-span-2">
                  <PerformanceTable />
                </div>
              </div>
            </section>

            {/* Secondary Analytics Section */}
            <section className="space-y-6">
              <div className="flex items-center gap-3 mb-6">
                <div className="w-1 h-6 bg-gradient-to-b from-purple-500 to-purple-400 rounded-full"></div>
                <h2 className="text-lg font-semibold text-foreground">Additional Insights</h2>
              </div>
              
              <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                {/* ... your other cards (Top Shops, Recent Activity) ... 
                    These can be placeholder cards for now */}
                <div className="lg:col-span-2 h-64 rounded-xl border border-border/50 bg-gradient-to-br from-white to-gray-50/30 shadow-sm flex items-center justify-center">
                  <div className="text-center space-y-2">
                    <div className="w-12 h-12 rounded-full bg-muted mx-auto flex items-center justify-center">
                      <span className="text-2xl">📊</span>
                    </div>
                    <p className="text-sm text-muted-foreground">Additional charts coming soon</p>
                  </div>
                </div>
                
                <div className="h-64 rounded-xl border border-border/50 bg-gradient-to-br from-white to-gray-50/30 shadow-sm flex items-center justify-center">
                  <div className="text-center space-y-2">
                    <div className="w-12 h-12 rounded-full bg-muted mx-auto flex items-center justify-center">
                      <span className="text-2xl">📈</span>
                    </div>
                    <p className="text-sm text-muted-foreground">Recent Activity</p>
                  </div>
                </div>
              </div>
            </section>

            {/* Forecasting Section */}
            <section className="space-y-6 pb-8">
              <div className="flex items-center gap-3 mb-6">
                <div className="w-1 h-6 bg-gradient-to-b from-orange-500 to-orange-400 rounded-full"></div>
                <h2 className="text-lg font-semibold text-foreground">Predictive Analysis</h2>
              </div>
              
              <ForecastTable />
            </section>
            
          </div>
        </main>
      </div>

      {/* Insight Panel - Positioned as overlay/sidebar */}
      <InsightPanel />
    </div>
  );
}