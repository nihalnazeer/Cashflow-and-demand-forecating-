"use client";

import { useDashboardStore } from "../../lib/store";
import { Card, CardHeader, CardTitle, CardContent } from "../ui/Card";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "../ui/Table";
import { TrendingUpIcon, AwardIcon, ShoppingBagIcon, TagIcon, LoaderIcon } from "lucide-react";

// Helper object to make titles and headers dynamic
const fieldLabels = {
  shop_name: "Shop",
  item_category_name: "Category", 
  total_revenue: "Revenue",
  total_demand: "Demand",
};

// Icon mapping for different dimensions
const dimensionIcons = {
  shop_name: ShoppingBagIcon,
  item_category_name: TagIcon,
};

// Professional loading skeleton for table rows
const TableRowSkeleton = () => (
  <>
    {[...Array(5)].map((_, i) => (
      <TableRow key={i} className="animate-pulse">
        <TableCell>
          <div className="h-4 w-6 bg-muted rounded"></div>
        </TableCell>
        <TableCell>
          <div className="h-4 w-32 bg-muted rounded"></div>
        </TableCell>
        <TableCell className="text-right">
          <div className="h-4 w-20 bg-muted rounded ml-auto"></div>
        </TableCell>
      </TableRow>
    ))}
  </>
);

// Enhanced rank display with medal icons for top 3
const RankDisplay = ({ rank }) => {
  if (rank <= 3) {
    return (
      <div className="flex items-center gap-2">
        <div className={`w-6 h-6 rounded-full flex items-center justify-center text-xs font-bold ${
          rank === 1 ? 'bg-yellow-100 text-yellow-700 border border-yellow-200' :
          rank === 2 ? 'bg-gray-100 text-gray-700 border border-gray-200' :
          'bg-amber-100 text-amber-700 border border-amber-200'
        }`}>
          {rank}
        </div>
        {rank === 1 && <AwardIcon className="h-4 w-4 text-yellow-600" />}
      </div>
    );
  }
  return (
    <div className="w-6 h-6 rounded-full bg-muted flex items-center justify-center text-xs font-medium text-muted-foreground">
      {rank}
    </div>
  );
};

export default function PerformanceTable() {
  const { performanceData = [], isLoadingPerformance, filters = {} } = useDashboardStore();

  // Generate title and headers dynamically from filters
  const dimensionLabel = fieldLabels[filters?.performanceDimension] || "Dimension";
  const metricLabel = fieldLabels[filters?.performanceMetric] || "Metric";
  const tableTitle = `Top ${dimensionLabel}s by ${metricLabel}`;
  const isRevenue = filters?.performanceMetric === 'total_revenue';
  
  // Get appropriate icon
  const IconComponent = dimensionIcons[filters?.performanceDimension] || TrendingUpIcon;

  return (
    <Card className="lg:col-span-2 animate-slide-in border-0 shadow-lg bg-gradient-to-br from-white to-gray-50/30">
      <CardHeader className="pb-4">
        <div className="flex items-center justify-between">
          <CardTitle className="flex items-center gap-3">
            <div className="p-2 rounded-lg bg-primary/10 border border-primary/20">
              <IconComponent className="h-5 w-5 text-primary" />
            </div>
            <div>
              <h3 className="text-lg font-semibold text-foreground">{tableTitle}</h3>
              <p className="text-sm text-muted-foreground font-normal mt-1">
                Performance ranking by {metricLabel.toLowerCase()}
              </p>
            </div>
          </CardTitle>
          
          {!isLoadingPerformance && performanceData.length > 0 && (
            <div className="text-right">
              <div className="text-xs text-muted-foreground uppercase tracking-wide font-medium">
                Total Entries
              </div>
              <div className="text-2xl font-bold text-foreground">
                {performanceData.length}
              </div>
            </div>
          )}
        </div>
      </CardHeader>
      
      <CardContent className="px-0">
        <div className="relative overflow-hidden">
          <Table>
            <TableHeader>
              <TableRow className="border-t border-border/50">
                <TableHead className="w-16 pl-6 text-xs font-semibold text-muted-foreground uppercase tracking-wide">
                  Rank
                </TableHead>
                <TableHead className="text-xs font-semibold text-muted-foreground uppercase tracking-wide">
                  {dimensionLabel} Name
                </TableHead>
                <TableHead className="text-right pr-6 text-xs font-semibold text-muted-foreground uppercase tracking-wide">
                  Total {metricLabel}
                </TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {isLoadingPerformance ? (
                <TableRowSkeleton />
              ) : performanceData.length > 0 ? (
                performanceData.map((row, index) => (
                  <TableRow 
                    key={row.dimension} 
                    className="group hover:bg-muted/30 transition-colors duration-200 border-border/30"
                  >
                    <TableCell className="pl-6">
                      <RankDisplay rank={index + 1} />
                    </TableCell>
                    <TableCell className="py-4">
                      <div className="flex items-center gap-3">
                        <div className="w-8 h-8 rounded-full bg-gradient-to-br from-primary/20 to-primary/10 flex items-center justify-center border border-primary/20">
                          <span className="text-xs font-semibold text-primary">
                            {row.dimension.charAt(0).toUpperCase()}
                          </span>
                        </div>
                        <span className="font-medium text-foreground group-hover:text-primary transition-colors">
                          {row.dimension}
                        </span>
                      </div>
                    </TableCell>
                    <TableCell className="text-right pr-6">
                      <div className="flex flex-col items-end">
                        <span className="font-bold text-lg text-foreground">
                          {isRevenue && '$'}
                          {row.metric.toLocaleString(undefined, { maximumFractionDigits: 0 })}
                        </span>
                        {isRevenue && (
                          <span className="text-xs text-muted-foreground">
                            {row.metric >= 1000000 ? 
                              `${(row.metric / 1000000).toFixed(1)}M` : 
                              row.metric >= 1000 ? 
                                `${(row.metric / 1000).toFixed(1)}K` : 
                                row.metric
                            }
                          </span>
                        )}
                      </div>
                    </TableCell>
                  </TableRow>
                ))
              ) : (
                <TableRow>
                  <TableCell colSpan={3} className="text-center py-12">
                    <div className="flex flex-col items-center gap-3">
                      <div className="w-12 h-12 rounded-full bg-muted flex items-center justify-center">
                        <IconComponent className="h-6 w-6 text-muted-foreground" />
                      </div>
                      <div>
                        <p className="font-medium text-foreground">No data available</p>
                        <p className="text-sm text-muted-foreground">Try adjusting your filters</p>
                      </div>
                    </div>
                  </TableCell>
                </TableRow>
              )}
            </TableBody>
          </Table>
        </div>
      </CardContent>
    </Card>
  );
}