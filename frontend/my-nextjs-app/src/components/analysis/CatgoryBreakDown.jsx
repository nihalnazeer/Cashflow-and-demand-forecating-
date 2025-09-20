import { PieChart, Pie, Cell, ResponsiveContainer, Legend, Tooltip } from "recharts";
import { Card, CardHeader, CardTitle, CardContent } from "../ui/Card";
import { TagIcon } from "lucide-react";

const categoryData = [
  { name: "Electronics", value: 45, color: "#00d4aa", revenue: 387000 },
  { name: "Accessories", value: 25, color: "#0ea5e9", revenue: 215000 },
  { name: "Computers", value: 20, color: "#a855f7", revenue: 172000 },
  { name: "Mobile", value: 10, color: "#f97316", revenue: 86000 }
];

const CustomTooltip = ({ active, payload }) => {
  if (active && payload && payload.length) {
    const data = payload[0].payload;
    return (
      <div className="glass-card p-3 border border-border/50 rounded-lg shadow-lg">
        <p className="text-sm font-medium text-foreground mb-1">{data.name}</p>
        <p className="text-xs text-muted-foreground">
          Revenue: ${data.revenue.toLocaleString()}
        </p>
        <p className="text-xs text-muted-foreground">
          Share: {data.value}%
        </p>
      </div>
    );
  }
  return null;
};

export default function CategoryBreakdown() {
  return (
    <Card className="animate-slide-in">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <TagIcon className="h-5 w-5 text-primary" />
          Category Breakdown
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="h-64">
          <ResponsiveContainer width="100%" height="100%">
            <PieChart>
              <Pie
                data={categoryData}
                cx="50%"
                cy="50%"
                outerRadius={80}
                fill="#8884d8"
                dataKey="value"
                label={({ name, value }) => `${name} ${value}%`}
                labelLine={false}
              >
                {categoryData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.color} />
                ))}
              </Pie>
              <Tooltip content={<CustomTooltip />} />
            </PieChart>
          </ResponsiveContainer>
        </div>
        
        <div className="space-y-2 mt-4">
          {categoryData.map((category, index) => (
            <div key={index} className="flex items-center justify-between text-sm">
              <div className="flex items-center gap-2">
                <div 
                  className="w-3 h-3 rounded-full" 
                  style={{ backgroundColor: category.color }}
                />
                <span className="text-muted-foreground">{category.name}</span>
              </div>
              <div className="text-right">
                <p className="font-medium text-foreground">
                  ${category.revenue.toLocaleString()}
                </p>
                <p className="text-xs text-muted-foreground">{category.value}%</p>
              </div>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
}