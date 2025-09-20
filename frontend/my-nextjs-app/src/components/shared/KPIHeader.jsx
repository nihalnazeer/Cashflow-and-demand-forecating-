"use client";

import { useDashboardStore } from "../../lib/store";
import { Card, CardContent } from "../ui/Card";
import { DollarSignIcon, ShoppingBagIcon, BarChartIcon } from "lucide-react";

// Helper to format large numbers
const formatCompactNumber = (number) => {
  if (typeof number !== 'number' || isNaN(number)) return "--";
  return new Intl.NumberFormat('en-US', {
    notation: 'compact',
    compactDisplay: 'short'
  }).format(number);
};

const KPICard = ({ title, value, icon: Icon, valuePrefix = "" }) => (
  <Card>
    <CardContent className="p-4">
      <div className="flex items-center justify-between mb-1">
        <p className="text-sm text-muted-foreground">{title}</p>
        <Icon className="h-4 w-4 text-muted-foreground" />
      </div>
      <p className="text-2xl font-bold text-foreground">
        {valuePrefix}{value}
      </p>
    </CardContent>
  </Card>
);

export default function KPIHeader() {
  // ✨ Get the live KPI data and loading state from the store
  const { kpis, isLoadingKpis } = useDashboardStore();

  const kpiItems = [
    {
      title: "Total Revenue",
      value: isLoadingKpis ? "..." : formatCompactNumber(kpis.total_revenue),
      icon: DollarSignIcon,
      valuePrefix: "$",
    },
    {
      title: "Total Demand",
      value: isLoadingKpis ? "..." : formatCompactNumber(kpis.total_demand),
      icon: ShoppingBagIcon,
    },
    {
      title: "Avg. Revenue per Item",
      value: isLoadingKpis || !kpis.total_demand ? "..." : (kpis.total_revenue / kpis.total_demand).toFixed(2),
      icon: BarChartIcon,
      valuePrefix: "$",
    },
    // You can add more KPIs here as your API evolves
  ];

  return (
    <div className="grid grid-cols-2 md:grid-cols-4 gap-6 animate-slide-in">
      {kpiItems.map(item => (
        <KPICard key={item.title} {...item} />
      ))}
    </div>
  );
}