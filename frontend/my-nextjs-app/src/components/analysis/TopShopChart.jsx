// components/analysis/TopShopsChart.jsx
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from "recharts";
import { useDashboardStore } from "../../lib/store";
import { Card, CardHeader, CardTitle, CardContent } from "../ui/Card";
import { MapPinIcon, TrendingUpIcon } from "lucide-react";

const CustomTooltip = ({ active, payload, label }) => {
  if (active && payload && payload.length) {
    return (
      <div className="glass-card p-3 border border-border/50 rounded-lg shadow-lg">
        <p className="text-sm font-medium text-foreground mb-2">{label}</p>
        <div className="space-y-1">
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded-full bg-chart-green" />
            <span className="text-xs text-muted-foreground">Revenue:</span>
            <span className="text-xs font-medium text-foreground">
              ${payload[0]?.value?.toLocaleString()}
            </span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded-full bg-chart-blue" />
            <span className="text-xs text-muted-foreground">Orders:</span>
            <span className="text-xs font-medium text-foreground">
              {payload[1]?.value?.toLocaleString()}
            </span>
          </div>
        </div>
      </div>
    );
  }
  return null;
};

export default function TopShopsChart() {
  const { topShops } = useDashboardStore();

  return (
    <Card className="animate-slide-in">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <MapPinIcon className="h-5 w-5 text-primary" />
          Top Performing Shops
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="h-64 mb-4">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={topShops} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" opacity={0.3} />
              <XAxis 
                dataKey="name" 
                stroke="hsl(var(--muted-foreground))"
                fontSize={12}
              />
              <YAxis 
                stroke="hsl(var(--muted-foreground))"
                fontSize={12}
                tickFormatter={(value) => `$${(value / 1000).toFixed(0)}K`}
              />
              <Tooltip content={<CustomTooltip />} />
              <Bar dataKey="revenue" fill="#00d4aa" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
        
        <div className="space-y-3">
          {topShops.slice(0, 3).map((shop, index) => (
            <div key={shop.id} className="flex items-center justify-between p-2 rounded-lg bg-muted/30">
              <div className="flex items-center gap-3">
                <span className={`text-xs font-bold px-2 py-1 rounded-full ${
                  index === 0 ? 'bg-yellow-500/20 text-yellow-400' : 
                  index === 1 ? 'bg-gray-500/20 text-gray-400' : 
                  'bg-orange-500/20 text-orange-400'
                }`}>
                  #{index + 1}
                </span>
                <span className="text-sm font-medium text-foreground">{shop.name}</span>
              </div>
              <div className="flex items-center gap-2">
                <TrendingUpIcon className="h-3 w-3 text-chart-green" />
                <span className="text-xs text-chart-green">+{shop.growth}%</span>
              </div>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
}