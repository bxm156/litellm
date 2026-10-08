import type { DailyActivityUserPageResponse, UserActivityRow } from "../dailyActivityApi";

export type FetchUserPage = (offset: number, limit: number) => Promise<DailyActivityUserPageResponse>;

export const NO_USER_KEY = "__no_user__";

export const userRowKey = (row: UserActivityRow): string => row.user_id ?? NO_USER_KEY;

export const userPrimaryLabel = (row: UserActivityRow): string => {
  const friendlyName = row.user_email || row.user_alias;
  return friendlyName || row.user_id || "(no user)";
};

export const userSecondaryLabel = (row: UserActivityRow): string | null => {
  const hasFriendlyName = Boolean(row.user_email || row.user_alias);
  return hasFriendlyName && row.user_id ? row.user_id : null;
};
