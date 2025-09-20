import { Card, CardContent } from "../ui/Card";
import { 
  DollarSignIcon, 
  ShoppingCartIcon, 
  TrendingUpIcon,
  UsersIcon,
  PercentIcon,
  ClockIcon
} from "lucide-react";

const quickStatsData = [
  {
    label: "Today's Revenue",
    value: "$12,847",
    change: "+8.2%",
    trend: "up",
    icon: DollarSignIcon,
    color: "text-chart-green"
  },
  {
    label: "Today's Orders",
    value: "94",
    change: "+12.5%",
    trend: "up",
    icon: ShoppingCartIcon,
    color: "text-chart-blue"
  },
  {
    label: "Active Users",
    value: "1,247",
    change: "+5.7%",
    trend: "up",
    icon: UsersIcon,
    color: "text-chart-purple"
  },
  {
    label: "Conversion Rate",
    value: "3.24%",
    change: "+0.8%",
    trend: "up",
    icon: PercentIcon,
    color: "text-chart-orange"
  }
];

export default function QuickStats() {
  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
      {quickStatsData.map((stat, index) => {
        const Icon = stat.icon;
        return (
          <Card key={index} className="glass-card hover:bg-card/90 transition-all duration-300">
            <CardContent className="p-4">
              <div className="flex items-center justify-between mb-2">
                <Icon className={`h-5 w-5 ${stat.color}`} />
                <span className={`text-xs font-medium flex items-center gap-1 ${
                  stat.trend === "up" ? "text-chart-green" : "text-chart-red"
                }`}>
                  <TrendingUpIcon className="h-3 w-3" />
                  {stat.change}
                </span>
              </div>
              <div className="space-y-1">
                <p className="text-2xl font-bold text-foreground">{stat.value}</p>
                <p className="text-xs text-muted-foreground">{stat.label}</p>
              </div>
            </CardContent>
          </Card>
        );
      })}
    </div>
  );
}
