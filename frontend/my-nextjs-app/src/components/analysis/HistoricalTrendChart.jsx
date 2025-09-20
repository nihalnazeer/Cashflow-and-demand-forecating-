import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from "recharts";
import { useDashboardStore } from "../../lib/store";
import { Card, CardHeader, CardTitle, CardContent } from "../ui/Card";
import { TrendingUpIcon, TrendingDownIcon, ActivityIcon, BarChartIcon } from "lucide-react";

// Helper function to format the date for the X-axis
const formatXAxisDate = (dateString) => {
  const date = new Date(dateString);
  return date.toLocaleDateString('en-US', { month: 'short', year: '2-digit' });
};

// A more polished Custom Tooltip component
const CustomTooltip = ({ active, payload, label }) => {
  if (active && payload && payload.length) {
    const date = new Date(label);
    const formattedDate = date.toLocaleDateString('en-US', { month: 'long', year: 'numeric' });
    
    return (
      <div className="rounded-lg border bg-background p-2 shadow-sm">
        <div className="grid grid-cols-2 gap-2">
          <div className="flex flex-col space-y-1">
            <span className="text-[0.70rem] uppercase text-muted-foreground">
              {formattedDate}
            </span>
            <span className="font-bold text-foreground">
              ${payload[0].value.toLocaleString()}
            </span>
          </div>
        </div>
      </div>
    );
  }
  return null;
};

// A simple skeleton loader for the chart
const ChartSkeleton = () => (
  <Card className="col-span-2">
    <CardHeader>
      <div className="h-6 w-48 bg-muted rounded-md animate-pulse" />
      <div className="h-4 w-64 bg-muted rounded-md animate-pulse mt-2" />
    </CardHeader>
    <CardContent>
      <div className="h-[350px] w-full bg-muted rounded-md animate-pulse" />
    </CardContent>
  </Card>
);


export default function HistoricalTrendChart() {
  const { historicalTrend, isLoadingTrend } = useDashboardStore();

  // ✨ Professional loading state with a skeleton component
  if (isLoadingTrend) {
    return <ChartSkeleton />;
  }

  // ✨ Professional empty state when no data is available
  if (!isLoadingTrend && historicalTrend.length === 0) {
    return (
      <Card className="col-span-2 flex flex-col items-center justify-center h-full min-h-[400px]">
        <BarChartIcon className="h-12 w-12 text-muted-foreground mb-4" />
        <h3 className="text-lg font-medium">No Data Available</h3>
        <p className="text-sm text-muted-foreground">The historical revenue trend could not be loaded.</p>
      </Card>
    );
  }

  const totalRevenue = historicalTrend.reduce((sum, item) => sum + item.revenue, 0);
  const currentMonthRevenue = historicalTrend[historicalTrend.length - 1]?.revenue || 0;
  const previousMonthRevenue = historicalTrend[historicalTrend.length - 2]?.revenue || 0;
  const revenueChange = previousMonthRevenue > 0 ? ((currentMonthRevenue - previousMonthRevenue) / previousMonthRevenue * 100) : 0;
  
  return (
    <Card className="col-span-2 animate-slide-in">
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="flex items-center gap-2">
              <ActivityIcon className="h-5 w-5 text-primary" />
              Overall Revenue Trend
            </CardTitle>
            <p className="text-sm text-muted-foreground mt-1">
              Live monthly performance from the database
            </p>
          </div>
          {historicalTrend.length > 1 && (
            <div className="text-right">
              <p className="text-2xl font-bold text-foreground">
                ${totalRevenue.toLocaleString(undefined, { maximumFractionDigits: 0 })}
              </p>
              <div className="flex items-center justify-end gap-1 text-sm">
                <span className={revenueChange >= 0 ? "text-chart-green" : "text-chart-red"}>
                   {revenueChange >= 0 ? (
                    <TrendingUpIcon className="h-4 w-4" />
                  ) : (
                    <TrendingDownIcon className="h-4 w-4" />
                  )}
                  {Math.abs(revenueChange).toFixed(1)}%
                </span>
                <span className="text-muted-foreground">vs last month</span>
              </div>
            </div>
          )}
        </div>
      </CardHeader>
      
      <CardContent>
        <div className="h-[350px]">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart 
              data={historicalTrend}
              margin={{ top: 5, right: 10, left: -20, bottom: 0 }}
            >
              <defs>
                {/* ✨ Using CSS variables for theme-aware colors */}
                <linearGradient id="revenueGradient" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="var(--color-primary)" stopOpacity={0.4}/>
                  <stop offset="95%" stopColor="var(--color-primary)" stopOpacity={0.05}/>
                </linearGradient>
              </defs>
              {/* ✨ Cleaner grid with no vertical lines */}
              <CartesianGrid 
                vertical={false}
                stroke="hsl(var(--border))" 
                strokeDasharray="3 3"
              />
              <XAxis 
                dataKey="month" 
                stroke="hsl(var(--muted-foreground))"
                axisLine={false}
                tickLine={false}
                fontSize={12}
                // ✨ Using the date formatter function
                tickFormatter={formatXAxisDate}
              />
              <YAxis 
                stroke="hsl(var(--muted-foreground))"
                axisLine={false}
                tickLine={false}
                fontSize={12}
                tickFormatter={(value) => `$${(value / 1000000).toFixed(0)}M`}
              />
              <Tooltip 
                cursor={{ stroke: 'hsl(var(--primary))', strokeWidth: 1, strokeDasharray: '3 3' }}
                content={<CustomTooltip />} 
              />
              <Area
                type="monotone"
                dataKey="revenue"
                stroke="hsl(var(--primary))"
                strokeWidth={2}
                fill="url(#revenueGradient)"
                name="Revenue"
                // ✨ Custom dot on hover for a better look
                activeDot={{ r: 6, strokeWidth: 2, fill: 'hsl(var(--primary))' }}
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </CardContent>
    </Card>
  );
}